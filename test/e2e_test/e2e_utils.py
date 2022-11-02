class ProfilingMetrics:
    def __init__(self, batch_size=-1, samples_per_second=0, peak_usage_bytes=None):
        self.batch_size = batch_size
        self.samples_per_second = samples_per_second
        self.peak_usage_bytes = peak_usage_bytes


class BaselineMetrics:
    def __init__(self, model_name, entry_point, given_batch_size, baseline_metric_list):
        self.model_name = model_name
        self.entry_point = entry_point
        self.given_batch_size = given_batch_size
        self.baseline_metric_list = baseline_metric_list

