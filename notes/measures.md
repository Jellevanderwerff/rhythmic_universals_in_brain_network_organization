# Behavioural measures used

## _G_ measure
The _G_ measure is a measure of grammatical redundancy that stems from information theory. Here, 'grammar' should be understood as all the different 'symbols' in a subject's (musical/linguistic) lexicon. 'Redundancy' is the number of symbols in the grammar that are strictly speaking unnecessary for transmitting the message. Such redundancy is a property of an organized system: it is the effect of (cultural) transmission from generation to generation, where combinations of symbols become standardized. To give an example from language: the 'rule' that the letter u always follows q makes the combination of symbols 'q' and 'u' redundant in the sense that two symbols could have been replaced by one new symbol. A musical example would be when, for instance, an eighth note always follows a quarternote. More redundancy (higher _G_) means more standardized combinations in the grammar as a result of (cultural) evolution. _G_ is very much related to entropy, and is calculated in much the same way. A key difference is that _G_ is the amount of redundancy in the entire grammar, while entropy is a value of the amount of information in one signal (here, one trial).

### How was it calculated?
- We converted the total collection of inter-onset intervals (IOIs) in a subject's grammar (done separately for each of the four conditions) into _k_ 'symbols' using K-means clustering (following the method in Ravignani et al., 2017). In the case of $k=3$, these symbols would represent 'short', 'medium', and 'long' intervals. The optimal number of clusters was determined by finding the lowest Silhouette score for a clustering with a minimum _k_ of 2 and a maximum _k_ of 8.
- For each combination of symbols in a trial, e.g. short-short-long, frequencies were counted across the subject's grammar, and converted into probabilities.
- These transition probabilities were then compared to those of a completely unstructed grammar with uniform transition probabilities, i.e. where each symbol can follow each other symbol with equal probability to calculate _G_.


### References
Jamieson & Mewhort, 2005
Ravignani et al., 2017


## Entropy
Entropy represents the amount of information in a signal, where lower entropy means a smaller amount of information, and higher entropy means a larger amount of information. Here, this signal is a rhythm. Entropy is based on probabilities of occurrence of different symbols (for us, note durations). It is best understood as the amount of information necessary to convey a message. Say we have a rhythm with 4 quarternotes, then the probability of a quarternote occurring is 1, and the set of different durations in the signal is 1. This rhythm has very low entropy (in fact, zero). Note how we only consider the different durations that occur in the signal, and not the different durations that could theoretically exist. Say now we have a rhythm with 6 notes that all have a different duration. If we add one note, and we want to predict what its duration will be based on these probabilities (six probabilities of 1/6), we have a very small chance of guessing its duration correctly. Therefore, entropy of this rhythm is high: we would need a lot of information to explain what this signal looks like if we want to transmit it correctly.


### How was it calculated?
- First, the tapping responses were quantized, where it was assumed that the smallest intended note duration was a sixteenth note. This was done using a Fourier transform. Here we first took the peak with the highest power (e.g. 397 ms). To understand what this value represents in terms of 'note durations', this value was compared to the duration of a theoretical quarternote, eighth note, or sixteenth note, based on the stimulus tempo. Following up on the example, for rhythms in the fast condition (tempo = 400 ms), 397 ms would be understood to be a quarternote, because it is closer to 400 ms than to a theoretical eighth note of 200 ms, and also closer than to a theoretical sixteenth note of 100 ms.
- Then, entropy was calculated for each trial based on the frequencies of occurrence of the note durations in the signal.


## Sum of normalized absolute asynchronies
This measure gives the total amount of stimulus-response asynchrony after tempo normalization. Remember that this was not a synchronization experiment, but that participants copied the stimulus.



### How was it calculated?
- We first tempo normalized the response sequence to the stimulus sequence, such that both are of equal duration.
- Then, we normalized the stimulus and response such that 1 represents the total duration of the stimulus and response.
- Then, for each pair of stimulus and response onsets, the absolute time difference was taken.
- Then, for each trial, these asynchronies were summed.
- In the final pp measures, the average was taken within condition.


## Number of small integer ratios introduced
Interval ratios represent the relationship between the durations of subsequent notes. If a rhythm consists of two quarternotes, the relationship of the first note to the combined duration of the two notes is 0.5. Note how for each rhythm of n _intervals_ we thus get _n_-1 interval ratios. The ratio is always represented as a number between 0 and 1. It is calculated as $ratio_k = \frac{interval_k}{interval_k + interval_{k+1}}$.

We followed the method from Roeske et al., by dividing the distribution of measured interval ratios into on-ratio and off-ratio values. This was done by equally dividing up the distribution into adjacent bins, where some are consideren on-ratio, and some off-ratio (see Methods section from Roeske for details of which ratios are considered on- and off-ratio).


### How was it calculated?
- We first tempo normalized the response to the stimulus (i.e. making the response of the same duration as the stimulus).
- We then calculated the interval ratios for both the stimulus and response.
- Then, for each trial we counted the number of integer ratios that were isochronous, binary, or ternary, and counted in the stimulus and in the response how many of those there were.
- We then subtracted the number of isochronous, binary, and ternary ratios in response from the number that was in the stimulus.
- As such, a negative value for the number of small integer ratios introduced means a reduction in the number of small integer ratios from stimulus to response (i.e. there were small integer ratios in the stimulus, but the pp did not copy those). A positive value means participants introduced small integer ratios in the response that weren't there in the stimulus.

### References
Roeske et al., 2017

## Tapping variability
Simply calculated as the difference between the coefficient of variation of the ITIs in the response, and the coefficient of variation of the IOIs in the stimulus. For the pp measures, averages were taken. It's always response - stimulus, so negative values met less variation in the response than in the stimulus.

## Tempo deviation
This is the ratio between the response tempo and the stimulus tempo, calculated as the sum of the response ITIs divided by the sum of the stimulus IOIs. Values below 1 thus mean that participant tapped faster than the stimulus. Values above 1 mean the participant tapped slower than the stimulus.

For the participant measures, this was slightly adjusted, and was calculated as the absolute difference between 1 and the ratio mentioned above. This because we are not interested in wheter participants speeded up or slowed down, but rather want a general measure of how 'inaccurate' participants were in terms of copying the tempo from stimulus to response.
