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
Entropy represents the amount of information in a signal, where lower entropy means a smaller amount of information, and higher entropy means a larger amount of information. Here, this signal is a rhythm. Entropy is based on probabilities of occurrence of different symbols (for us, note durations). It is best understood as the amount of information necessary to convey a message. Say we have a rhythm with 4 quarternotes. The probability of a quarternote occurring is 1, and the set of different durations in the signal is 1. This rhythm has very low entropy (in fact, zero). Note how we only consider the different durations that occur in the signal, and not the different durations that could theoretically exist. Say now we have a rhythm with 6 notes that all have a different duration. If we add one note, and we want to predict what its duration will be based on these probabilities (six probabilities of 1/6), we have a small chance of guessing its duration correctly. Therefore, entropy of this rhythm is high: we would need a lot of information to explain what this signal looks like if we want to transmit it correctly.


### How was it calculated?
- First, the tapping responses were quantized, where it was assumed that the smallest intended note duration was a sixteenth note. This was done using a Fourier transform. Here we first took the peak with the highest power (e.g. 397 ms). To understand what this value represents in terms of 'note durations', this value was compared to the duration of a theoretical quarternote, eighth note, or sixteenth note, based on the stimulus tempo. Following up on the example, for rhythms in the fast condition (tempo = 400 ms), 397 ms would be understood to be a quarternote, because it is closer to 400 ms than to a theoretical eighth note of 200 ms, and also closer than to a theoretical sixteenth note of 100 ms.
- Then, entropy was calculated for each trial based on the frequencies of occurrence of the note durations in the signal.


## Edit (or Levenshtein) distance
Levenshtein distance is calculated between two strings as the number of insertions, deletions, or substitutions necessary to transform the one string into the other. Say we have two strings, 'hello' and 'hlelo'. We need only one substitution to change the one string into the other, and so the edit distance is one. For rhythms, it is less obvious how one would calculate this, as how would we represent the rhythm as a string? This is generally done by representing an 'underlying grid' as a string of zeroes, and placing each note onset at its appropriate position on the grid as a one. So, a rhythm with a total duration of one quarternote could be represented as a string of four zeroes, assuming the smallest to-be-expected note duration is a sixteenth note. Say this rhythm then also consists of only one quarternote, then the final string is '1000'.



### How was it calculated?
- We first quantized the rhythm (see under 'entropy') to sixteenth notes.
- We then represented the rhythm as a string of zeroes and ones, with an underlying grid of sixteenth notes.
- We calculated the Levenshtein distance between to strings.


## Preference for binary or ternary interval ratios
Interval ratios represent the relationship between the durations of subsequent notes. If a rhythm consists of two quarternotes, the relationship of the first note to the combined duration of the two notes is 0.5. Note how for each rhythm of n _intervals_ we thus get _n_-1 interval ratios. The ratio is always represented as a number between 0 and 1. It is calculated as $ratio_k = \frac{interval_k}{interval_k + interval_{k+1}}$.

Now, to investigate whether people more frequently produce two subsequent notes with a binary relationship (e.g. quarternote-halfnote) or with a ternary relationship (e.g. eighth note-dotted quarternote), using the interval ratios we counted the number of occurrences of binary and ternary relationships. However, a problem here arises, because produced rhythms are seldom (if ever) perfectly produced, and say we want to count the number of occurrences of interval ratios 0.25 and 0.75 (i.e. ternary relationships), we would have to specify a bandwith around those numbers, and it is difficult to say what that bandwith should be (usually, people just assume some numbers). So, in order to have a measure that is the most objective, and requires the fewest assumptions, we first quantized the rhythms to sixteenth notes using the procedure described under 'entropy', and only then calculated the interval ratios.

### How was it calculated?
- We first quantized the rhythm (see under 'entropy') to sixteenth notes.
- We then calculated the interval ratios between dyads of intervals using the formula described above.
- We then calculated the proportion between binary ratios and the total number of ratios, and the proportion of ternary ratios and the total number of ratios.

### References
Roeske et al., 2017



