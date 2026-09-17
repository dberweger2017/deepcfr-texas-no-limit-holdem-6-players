# Card-diversity runtime amendment

This branch changes only the admission runtime caps for the already merged
card-diversity protocol. `max_reference_seconds` increases from 3,600 to 5,400
seconds and `max_fit_seconds` from 1,200 to 1,800 seconds. The scientific
matrix, board generator, shared range, seeds, optimizer updates, validation and
test thresholds, selection logic, and promotion policy are unchanged.

The amendment responds to the two preserved admission failures documented in
[`holdem-card-diversity-admission.md`](holdem-card-diversity-admission.md) and
the later idle calibration estimate (2,807.98 seconds reference and 1,203.58
seconds fitting, each including the existing 1.5 allowance). It authorizes one
bounded local campaign with no rental and no adaptive sweeps. The campaign
must retain partial failures and stop at these caps.
