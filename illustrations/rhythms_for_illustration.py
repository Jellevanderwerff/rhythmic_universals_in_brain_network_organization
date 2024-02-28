from thebeat.music import Rhythm
import os

out_dir = os.path.join("illustrations", "example_rhythms")

# Preference for isochronous
r = Rhythm.from_note_values([4, 4, 4, 4])
r.plot_rhythm(out_dir + "/isoc_1.pdf", print_staff=False)
r = Rhythm.from_note_values([8, 8, 8, 8], time_signature=(2, 4))
r.plot_rhythm(out_dir + "/isoc_2.pdf", print_staff=False)

# Preference for binary, ternary
r = Rhythm.from_note_values([8, 4, 8, 4], time_signature=(3, 4))
r.plot_rhythm(out_dir + "/binary_1.pdf", print_staff=False)
r = Rhythm.from_note_values([4, 2, 4, 2], time_signature=(6, 4))
r.plot_rhythm(out_dir + "/binary_2.pdf", print_staff=False)

r = Rhythm.from_note_values([16, 4, 16, 4, 16, 4, 16, 4], time_signature=(5, 4))
r.plot_rhythm(out_dir + "/ternary_1.pdf", print_staff=False)
r = Rhythm.from_note_values([8, 2, 8, 2, 8, 2, 8, 2], time_signature=(10, 4))
r.plot_rhythm(out_dir + "/ternary_2.pdf", print_staff=False)


# Preference for durational categories
r = Rhythm.from_note_values([16, 16, 16, 16, 4], time_signature=(2, 4))
r.plot_rhythm(out_dir + "/dur_cat_1.pdf", print_staff=False)
r = Rhythm.from_note_values([4, 8, 8, 8, 8], time_signature=(3, 4))
r.plot_rhythm(out_dir + "/dur_cat_2.pdf", print_staff=False)

# Preference for grammatical motifs
r = Rhythm.from_note_values([8, 8, 4, 8, 8, 4], time_signature=(4, 4))
r.plot_rhythm(out_dir + "/gram_motif_1.pdf", print_staff=False)
r = Rhythm([250, 250, 250, 1250])
r.plot_rhythm(out_dir + "/gram_motif_2.pdf", print_staff=False)
