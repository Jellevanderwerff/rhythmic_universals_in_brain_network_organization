from thebeat.music import Rhythm
import random
import os

for i in range(4):
    r = Rhythm.generate_random_rhythm(allowed_note_values=[4, 8, 16], n_rests=random.randint(2, 4))
    r.plot_rhythm(os.path.join("illustrations", "example_rhythms", f"random_rhythm_{i}.pdf"), print_staff=False)
