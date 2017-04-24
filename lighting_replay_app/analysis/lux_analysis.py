"""
Analyze lux sensor data and use it to predict future behavior of the light.
"""
import math
from numpy.core.numeric import NaN
from collections import deque
from sklearn.preprocessing import StandardScaler

from thingflow.base import OutputThing, FunctionFilter,\
                           SensorEvent, filtermethod
from thingflow.filters.transducer import SensorSlidingMean

from lux_time_utils import get_sunrise_sunset, time_of_day_to_zone,\
    NUM_ZONES, dt_to_minutes, minutes_to_time

MAX_TIME_INTERVAL = 60*4
EXPECTED_TIME_INTERVAL = 60


@filtermethod(OutputThing)
def fill_in_missing_times(this):
    def on_next(self, x):
        if (self.last_time is not None) and \
           (x.ts - self.last_time)>MAX_TIME_INTERVAL:
            ts = self.last_time + EXPECTED_TIME_INTERVAL
            missing = 0
            while (x.ts-ts)>EXPECTED_TIME_INTERVAL:
                if missing==0:
                    self._dispatch_next(SensorEvent(sensor_id=x.sensor_id,
                                                    ts=ts, val=NaN))
                ts += EXPECTED_TIME_INTERVAL
                missing += 1
            print("Found %s missing samples" % missing)
        self.last_time = x.ts
        self._dispatch_next(x)
    
    f = FunctionFilter(this, on_next=on_next,
                       name="fill_in_missing_times()")
    setattr(f, 'last_time', None)
    return f

class SensorSlidingMeanPassNaNs(SensorSlidingMean):
    """Variant of SensorSlidingMean that passes on NaN-valued events without
    including them in the history. We clear the sliding window each time
    an NaN is found.
    """
    def __init__(self, history_samples):
        super().__init__(history_samples)
        
    def step(self, event):
        if math.isnan(event.val):
            if len(self.history)>0:
                #self.state -= self.history.popleft().val
                self.state = None
                self.history = deque(maxlen=self.history_samples)
            return event
        else:
            return SensorSlidingMean.step(self, event)
        
class CaptureNaNIndexes:
    """AntEvents subscriber that watches for values that are
    NaN and tracks those indexes.
    """
    def __init__(self):
        self.idx = 0
        self.nan_indexes = []

    def on_next(self, x):
        if math.isnan(x.val):
            self.nan_indexes.append(self.idx)
        self.idx += 1

    def on_error(self, e):
        pass

    def on_completed(self):
        pass

    def replace_nans(self, array, val):
        for idx in self.nan_indexes:
            print("replacing index %s with %s" % (idx, val))
            array[idx] = val

    def new_array_replace_nans(self, array, replace_value):
        result = []
        for (idx, val) in enumerate(array):
            if idx in self.nan_indexes:
                result.append(replace_value)
            else:
                result.append(val)
        return result
    

class HmmScanner:
    """Scan through a set of samples in sequential order and
    build up sequences for each time zone that can be passed to
    the HMM learn. For each zone, we need to pass the fit() method
    the concatenated sequence of samples and the lengths of each
    subsequence. We break the samples into multiple subsequences
    whenever we encounter a time gap (as indicated via a NaN value)
    or when we cross between zones.
    """
    def __init__(self):
        self.length = None
        self.zone = None
        self.samples_by_zone = [[] for zone in range(NUM_ZONES)]
        self.lengths_by_zone = [[] for zone in range(NUM_ZONES)]

    def _start_sequence(self, zone, s):
        self.length = 1
        self.zone = zone
        self.samples_by_zone[zone].append(s)
        
    def _complete_sequence(self):
        if self.length is not None:
            assert self.zone is not None
            self.lengths_by_zone[self.zone].append(self.length)
        self.zone = None
        self.length = None
        
    def process_samples(self, samples, timestamps):
        for (s, t) in zip(samples, timestamps):
            s = int(s) if not math.isnan(s) else NaN
            (sunrise, sunset) = get_sunrise_sunset(t.year, t.month, t.day)
            current_zone = time_of_day_to_zone(dt_to_minutes(t), sunrise,
                                               sunset)
            if self.length is None:
                if math.isnan(s):
                    continue
                else:
                    self._start_sequence(current_zone, s)
            elif math.isnan(s):
                self._complete_sequence()
            elif self.zone != current_zone:
                self._complete_sequence()
                self._start_sequence(current_zone, s)
            else: # just extend the current sequence
                self.length += 1
                self.samples_by_zone[self.zone].append(s)
        # see if there was an in-progress sequence for which we need to add
        # the length
        self._complete_sequence()
        # sanity check
        for zone in range(NUM_ZONES):
            assert sum(self.lengths_by_zone[zone])==len(self.samples_by_zone[zone])


BACKCHECK_LENGTH = 10

class ScanState:
    """A state machine for finding sequences of on or off samples
    """
    WAITING_FOR_TRANSITION = 0
    RECORDING_LENGTH = 1
    NAN_STATE = 2
    STATE_NAMES=['WAITING', 'RECORDING' 'NAN']

    def __init__(self):
        self.state = None
        self.prev_sample = None
        self.start_zone = None
        self.start_time = None
        self.prev_zone = None
        self.length = None
        self.recorded_events = None
        # The backcheck queue maintains a queue of older samples.
        # This is used to include the sample which is BACKCHECK_LENGTH
        # samples back, which is helpful for some machine learning algorihms.
        self.backcheck_queue = []

    def add_sample(self, s, t):
        s = int(s) if not math.isnan(s) else NaN
        (sunrise, sunset) = get_sunrise_sunset(t.year, t.month, t.day)
        current_zone = time_of_day_to_zone(dt_to_minutes(t), sunrise,
                                           sunset)
        if self.state==None:
            if math.isnan(s):
                self.state = ScanState.NAN_STATE
            else:
                self.state = ScanState.WAITING_FOR_TRANSITION
            print("initial state is %s" % ScanState.STATE_NAMES[self.state])
        elif math.isnan(s):
            if self.state!=ScanState.NAN_STATE:
                print("changing to NAN_STATE at %s" % t)
            self.state = ScanState.NAN_STATE
            self.length = None
            self.start_zone = None
            self.start_time = None
            self.prev_zone = None
            self.recorded_events = None
            self.backcheck_queue = []
        elif self.state==ScanState.NAN_STATE:
            print("changing to WAITING_STATE at %s" % t)
            self.state = ScanState.WAITING_FOR_TRANSITION # got a value
        elif self.state==ScanState.WAITING_FOR_TRANSITION and (s!=self.prev_sample):
            print("changing to RECORDING_STATE(%s) at %s" % (s,t))
            self.state = ScanState.RECORDING_LENGTH
            self.start_zone = current_zone
            self.start_time = t
            self.length = 1
            back = self.backcheck_queue[0] if len(self.backcheck_queue)>0 else s
            self.recorded_events = [(s, t, current_zone, current_zone, 1, back),]
        elif self.state==ScanState.RECORDING_LENGTH:
            if s==self.prev_sample:
                self.length += 1
                back = self.backcheck_queue[0] if len(self.backcheck_queue)>0 else s
                self.recorded_events.append((s, t, self.start_zone, current_zone,
                                             self.length, back),)
            else:
                if self.prev_sample==0:
                    print("OFF sequence zone %s, length %d" % (self.start_zone,
                                                               self.length))
                    self.record_off_sequence(self.start_zone, self.start_time,
                                             self.prev_zone,
                                             self.length)
                else:
                    print("ON sequence zone %s, length %d" % (self.start_zone,
                                                              self.length))
                    self.record_on_sequence(self.start_zone, self.start_time,
                                            self.prev_zone,
                                            self.length)
                # we know this was valid, so call record_event() for each event
                for (evt_s, evt_dt, evt_start_zone, evt_current_zone,
                     evt_length, evt_back) in self.recorded_events:
                    self.record_event(evt_s, evt_dt, evt_start_zone, evt_current_zone,
                                      evt_length, evt_back)
                # reset for the new value of s
                self.length = 1
                self.start_zone = current_zone
                self.start_time = t
                back = self.backcheck_queue[0] if len(self.backcheck_queue)>0 else s
                self.recorded_events = [(s, t, current_zone, current_zone, 1, back),]
        self.prev_sample = s
        self.prev_zone = current_zone
        if not math.isnan(s):
            self.backcheck_queue.append(s)
            if len(self.backcheck_queue)>BACKCHECK_LENGTH:
                self.backcheck_queue.pop(0) # remove the oldest

    def record_event(self, s, dt, start_zone, current_zone, current_length, back_event):
        """Template method that is called when we have a valid sample we
        can use. A sample is valid if it is preceeded by zero or more samples
        of the same value and one or more samples of the other value. This
        means we can provide a correct value for the length. If the samples of
        the current value were preceeded by a NaN, we don't know for certain
        how long that value was present.
        """
        pass

    def record_on_sequence(self, start_zone, start_time, end_zone, length):
        """Template method that is called when we have a sequence of on samples
        we can use. A sequence is valid if it has one or more samples of the
        same value, both preceeded and succeeded by one or more samples of the
        oppositive value. If there is an NaN on either side, we cannot
        conclusively determine the length.
        """
        pass

    def record_off_sequence(self, start_zone, start_time, end_zone, length):
        """Template method that is called when we have a sequence of off samples
        we can use.
        """
        pass

class LengthHistogramState(ScanState):
    """Build lists of on/off lengths that can be used to compute
    histograms.
    """
    def __init__(self):
        super().__init__()
        self.on_lengths = [[] for i in range(NUM_ZONES)]
        self.off_lengths = [[] for i in range(NUM_ZONES)]

    def record_on_sequence(self, start_zone, start_time, end_zone, length):
        self.on_lengths[start_zone].append(length)

    def record_off_sequence(self, start_zone, start_time, end_zone, length):
        self.off_lengths[start_zone].append(length)
        
def build_length_histogram_data(samples, timestamps):
    """Given a series of samples and timestamps, find on and off
    sequences and build lists of the sequence lengths per zone. A
    'sequence' is a series of samples of the same value preceeded by at
    last one sample of the opposite value (as oppposed to a break in the
    readings). Returns a LengthHistogramState object containing on_lengths
    and off_lengths members.
    """
    state = LengthHistogramState()
    for (s, t) in zip(samples, timestamps):
        state.add_sample(s, t)
    for zone in range(NUM_ZONES):
        state.on_lengths[zone].sort()
        state.off_lengths[zone].sort()
    return state

class LightPredictionStates(ScanState):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        
    def record_on_sequence(self, start_zone, start_time, end_zone, length):
        self.trainer.on_lengths[start_zone].append(length)
        self.trainer.on_lengths_with_start.append((dt_to_minutes(start_time), length),)

    def record_off_sequence(self, start_zone, start_time, end_zone, length):
        self.trainer.off_lengths[start_zone].append(length)
        self.trainer.off_lengths_with_start.append((dt_to_minutes(start_time), length),)

    def record_event(self, s, dt, start_zone, current_zone, current_length, back_event):
        length_bucket = int(round(current_length/BACKCHECK_LENGTH))+1
        if s==1:
            f = (dt.hour, current_zone, 0, length_bucket, back_event)
        else:
            f = (dt.hour, current_zone, length_bucket, 0, back_event)
        self.trainer.training_features.append(self.trainer.feature_filter(f))
        self.trainer.training_targets.append(s)
        self.trainer.obs_by_zone[current_zone].append(s)
        
        
class LightPredictionTrainer:
    """This class preprocessed the data to create features for machine
    learning.
    """
    def __init__(self, feature_filter=lambda x: x):
        """The feature filter is a function that can remove elements from
        the feature tuple. By default the tuple is
        (hour, zone, len_off, len_on, back_event).
        """
        self.on_lengths = [[] for i in range(NUM_ZONES)]
        self.off_lengths = [[] for i in range(NUM_ZONES)]
        self.on_lengths_with_start = []
        self.off_lengths_with_start = []
        self.training_features = []
        self.training_targets = []
        self.scaled_features = None
        self.scaler = None
        self.feature_filter = feature_filter
        self.obs_by_zone = [[] for z in range(NUM_ZONES)]
    
    def _compute_lengths(self, samples, timestamps):
        state_machine = LightPredictionStates(self)
        for (s, t) in zip(samples, timestamps):
            state_machine.add_sample(s, t)
        print("self.on_lengths")
        print(self.on_lengths)
        print("self.off_lengths")
        print(self.off_lengths)

    def _compute_scaled_features(self):
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.training_features)

    def _compute_zones(self):
        """Lets compute some zones where we have the same number of samples
        per zone.
        """
        self.on_lengths_with_start.sort()
        self.off_lengths_with_start.sort()
        samples_per_zone = int(round(len(self.on_lengths_with_start)/4))
        print("Compute zones: %d total samples, try for %d samples per zone" %
              (len(self.on_lengths_with_start), samples_per_zone))
        zone_boundaries = []
        prev_btime = 0
        for boundary_idx in [samples_per_zone, 2*samples_per_zone, 3*samples_per_zone]:
            (btime, _) = self.on_lengths_with_start[boundary_idx]
            while btime==prev_btime:
                # if the same time as the last keep going
                boundary_idx += 1
                (btime, _) = self.on_lengths_with_start[boundary_idx]
            zone_boundaries.append(btime)
        for i in zone_boundaries:
            (hr, mn) = minutes_to_time(i)
            print("computed boundary: %d minutes %02d:%02d" % (i, hr, mn))
                                                         
        
        
    def compute(self, samples, timestamps):
        self._compute_lengths(samples, timestamps)
        self._compute_scaled_features()
        self._compute_zones()
        
    def features_for_prediction(self, s, dt, length, sample_queue, scaled=True):
        minutes = dt_to_minutes(dt)
        (sunrise, sunset) = get_sunrise_sunset(dt.year, dt.month, dt.day)
        zone = time_of_day_to_zone(minutes, sunrise, sunset)
        length_bucket = int(round(length/BACKCHECK_LENGTH))+1
        hist_value = sample_queue[0] if len(sample_queue)>0 else s
        if len(sample_queue)>=BACKCHECK_LENGTH:
            sample_queue.pop(0)
        if s==1:
            raw = self.feature_filter((dt.hour, zone, 0, length_bucket, hist_value),)
        else:
            raw = self.feature_filter((dt.hour, zone, length_bucket, 0, hist_value),)
        if scaled:
            return self.scaler.transform([raw,])
        else:
            return [raw,]
        
    def sort_lengths(self):
        for lengths in [self.on_lengths, self.off_lengths]:
            for zone in range(NUM_ZONES):
                lengths[zone].sort()
        print("Sorted On Lengths: %s" % self.on_lengths)
        print("Sorted Off Lengths: %s" % self.off_lengths)


def lengths_to_histogram(lengths):
    sl = sorted(lengths)
    if len(sl)==0:
        return []
    max_len = sl[-1]
    hist = [0 for i in range(max_len+1)]
    for l in sl:
        hist[l] += 1
    return hist
 
#compute_lengths(kmeans_lux, smoothed_series_writer.index)
#ON_PROBS = compute_probs(kmeans_lux, smoothed_series_writer.index)

import random
random.seed()



class LightPredictorLengthOnly:
    """Use the length data to determine when the light should go on and off
    and how long each time.
    """
    def __init__(self, initial_state, trainer):
        self.current_state = initial_state
        self.current_length = None
        self.current_zone = None
        self.trainer = trainer

    def _choose_length(self, zone, state):
        if state==0:
            lengths = self.trainer.off_lengths
        else:
            lengths = self.trainer.on_lengths
        if len(lengths[zone])==0:
            orig_zone = zone
            while True:
                zone = (zone+1)%NUM_ZONES
                assert zone!=orig_zone
                if len(lengths[zone])>0:
                    print("No lengths for %s available for zone %s, using value from zone %s" %
                          ('ON' if state else 'Off', orig_zone, zone))
                    break
        base_length = random.choice(lengths[zone])
        # add some noise to the length, then subtract 1 since this step is already the first sample
        length = max(int(round(random.gauss(base_length, 0.1*base_length))), 1) - 1
        print("chose %s length %s, randomized to %s" %
              ('ON' if state else 'OFF', base_length, length+1))
        return length
    
    def predict(self, dt):
        (sunrise, sunset) = get_sunrise_sunset(dt.year, dt.month, dt.day)
        new_zone = time_of_day_to_zone(dt_to_minutes(dt), sunrise, sunset)
        if self.current_zone==None:
            self.current_zone = new_zone
        if self.current_length==None:
            self.current_length = self._choose_length(self.current_zone, self.current_state)
        elif self.current_length==0:
            self.current_state = 0 if self.current_state else 1
            self.current_length = self._choose_length(new_zone, self.current_state)
        else:
            self.current_length -= 1
        self.current_zone = new_zone
        return self.current_state        
