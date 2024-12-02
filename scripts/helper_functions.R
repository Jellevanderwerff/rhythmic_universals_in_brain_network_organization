calculatePDVs <- function(stim_onsets, resp_onsets) {

  n_responses <- length(resp_onsets)
  output <- double(n_responses)

  for (i in 1:n_responses) {

    if (i == 1 | i == 2) {
      stim_ioi_nmin2 <- NA
    } else {
      stim_ioi_nmin2 <- stim_onsets[i-1] - stim_onsets[i-2] # delta S_n
    }

    if (!is.na(resp_onsets[i]) ) {
      pdv <- (resp_onsets[i] - stim_onsets[i]) / stim_ioi_nmin2
    } else {
      pdv <- NA
    }

    output[i] <- pdv

  }

  return(output)

}


splitPythonListString <- function(string) {
  string <- str_remove(string, "\\[")
  string <- str_remove(string, "\\]")
  string <- str_split(string, ", ")
  string <- as.double(unlist(string))

  return(string)
}