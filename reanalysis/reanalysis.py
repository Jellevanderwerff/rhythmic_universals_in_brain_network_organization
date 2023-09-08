from repp.analysis import REPPAnalysis
from new_config import npor_new_config
import os


repp_analysis = REPPAnalysis(config=npor_new_config)


for file in os.listdir(os.path.join('reanalysis','incorrectly_analyzed')):
    if not file.endswith('.wav'):
        continue
    print(f"Analyzing {file}")
    recording_filepath = os.path.join('reanalysis','incorrectly_analyzed', file)
    _, extracted_onsets, analysis = repp_analysis.do_analysis_tapping_only(recording_filepath,
                                                                      title_plot='old config',
                                                                      output_plot=f"reanalysis/incorrectly_analyzed/plots/{file[:-4]}.png")
    print(extracted_onsets)
    print(analysis)


"""
for file in os.listdir(os.path.join('reanalysis','correctly_analyzed')):
    if not file.endswith('.wav'):
        continue
    print(f"Analyzing {file}")
    recording_filepath = os.path.join('reanalysis','correctly_analyzed', file)
    _, extracted_onsets, analysis = repp_analysis.do_analysis_tapping_only(recording_filepath,
                                                                      title_plot='old config',
                                                                      output_plot=f"reanalysis/correctly_analyzed/plots/{file[:-4]}.png")
    print(extracted_onsets)
    print(analysis)
"""