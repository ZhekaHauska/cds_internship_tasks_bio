import os
import numpy as np
from typing import Dict
import seaborn as sns
import imageio
import aim

EPS = 1e-24

class BaseLogger:
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        return

    def define_metric(self, *args, **kwargs):
        raise NotImplementedError

    def process_image(self, image):
        return image

    def process_video(self, video):
        return video

    def process_figure(self, figure):
        return figure

    def process_dist(self, dist):
        return dist

    @property
    def name(self):
        raise NotImplementedError


class AimLogger(BaseLogger):
    def __init__(self, config):
        self.run = aim.Run(
            experiment=config['run']['project_name'],
            repo=config['run']['repo'],
            log_system_params=True
        )
        print('[Aim] Running ', self.run.hash)
        tags = []
        if 'tags' in config['run']:
            t = config['run']['tags']
            if isinstance(t, list):
                tags.extend(t)
            elif t is not None:
                tags.append(t)
        for tag in tags:
            self.run.add_tag(tag)
        self.run['hparams'] = config

    def log(self, *args, **kwargs):
        self.run.track(*args, **kwargs)

    def close(self):
        print('[Aim] Closing ', self.run.hash)
        self.run.close()

    def define_metric(self, *args, **kwargs):
        ...

    def process_image(self, image):
        return aim.Image(image)

    def process_video(self, video):
        return aim.Image(video)

    def process_figure(self, figure):
        return aim.Image(figure.figure)

    def process_dist(self, dist):
        return aim.Image(sns.histplot(dist).figure)

    @property
    def name(self):
        return self.run.hash


class BaseMetric:
    logger: BaseLogger
    def __init__(self, logger: BaseLogger, runner,
                 update_step, log_step, update_period, log_period):
        self.logger = logger
        self.runner = runner
        self.update_step = update_step
        self.log_step = log_step
        self.update_period = update_period
        self.log_period = log_period

        self.last_update_step = None
        self.last_log_step = None

    def step(self):
        update_step = self.get_attr(self.update_step)
        log_step = self.get_attr(self.log_step)

        if (self.last_update_step is None) or (self.last_update_step != update_step):
            if (update_step % self.update_period) == 0:
                self.update()

        if (self.last_log_step is None) or (self.last_log_step != log_step):
            if (log_step % self.log_period) == 0:
                self.log(log_step)

        self.last_update_step = update_step
        self.last_log_step = log_step

    def update(self):
        raise NotImplementedError

    def log(self, step):
        raise NotImplementedError

    def get_attr(self, attr):
        obj = self.runner
        for a in attr.split('.'):
            obj = getattr(obj, a)
        return obj


class MetricsRack:
    metrics: Dict[str, BaseMetric]

    def __init__(self, logger, runner, **kwargs):
        self.metrics = dict()

        for name, params in kwargs.items():
            cls = params['class']
            params = params['params']
            self.metrics[name] = eval(cls)(**params, logger=logger, runner=runner)

    def step(self):
        for name in self.metrics.keys():
            self.metrics[name].step()


class ScalarMetrics(BaseMetric):
    def __init__(self, metrics, logger, runner,
                 update_step, log_step, update_period, log_period):
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)

        self.metrics = {metric: [] for metric in metrics.keys()}

        for metric in metrics.keys():
            self.logger.define_metric(metric, step_metric=self.log_step)

        self.agg_func = {
            metric: eval(params['agg']) if type(params['agg']) is str else params['agg']
            for metric, params in metrics.items()
        }
        self.att_to_log = {
            metric: params['att']
            for metric, params in metrics.items()
        }

    def update(self):
        for name in self.metrics.keys():
            value = self.get_attr(self.att_to_log[name])
            self.metrics[name].append(value)

    def log(self, step):
        log_dict = {self.log_step: step}
        log_dict.update(self._summarize())
        self.logger.log(log_dict)
        self._reset()

    def _reset(self):
        self.metrics = {metric: [] for metric in self.metrics.keys()}

    def _summarize(self):
        result = dict()
        for key, values in self.metrics.items():
            values = np.array(values).flatten()
            if len(values) == 0:
                continue
            filtered_values = values[~np.isnan(values)]
            if len(filtered_values) > 0:
                result[key] = self.agg_func[key](filtered_values)
        return result


class ImageMetrics(BaseMetric):
    def __init__(self, metrics, logger: BaseLogger, runner,
                 update_step, log_step, update_period, log_period,
                 log_fps, log_dir='/tmp'):
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)

        self.metrics = {metric: [] for metric in metrics}
        self.att_to_log = {
            metric: params['att']
            for metric, params in metrics.items()
        }
        self.logger = logger
        self.log_fps = log_fps
        self.log_dir = log_dir

    def update(self):
        for name in self.metrics.keys():
            value = self.get_attr(self.att_to_log[name])
            self.metrics[name].append(value)

    def log(self, step):
        log_dict = {self.log_step: step}
        for metric, values in self.metrics.items():
            if len(values) > 1:
                gif_path = os.path.join(
                    self.log_dir,
                    f'{self.logger.name}_{metric.split("/")[-1]}_{step}.gif'
                )
                # use new v3 API
                imageio.v3.imwrite(
                    # mode 'L': gray 8-bit ints; duration = 1000 / fps; loop == 0: infinitely
                    gif_path, values, mode='L', duration=1000/self.log_fps, loop=0
                )
                log_dict[metric] = self.logger.process_video(gif_path)
            elif len(values) == 1:
                log_dict[metric] = self.logger.process_image(values[0])

        self.logger.log(log_dict)
        self._reset()

    def _reset(self):
        self.metrics = {metric: [] for metric in self.metrics.keys()}
