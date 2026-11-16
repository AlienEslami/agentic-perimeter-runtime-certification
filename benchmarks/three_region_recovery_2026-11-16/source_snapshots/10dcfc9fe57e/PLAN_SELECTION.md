# Fixed plans for the certificate study

The first trajectory screen retains a base arrival rate of 6000 veh/h.
Increasing entry everywhere to 0.65 sends too much traffic into the congested
regions. Lower entries of 0.50 to Regions 2 and 3 retain the vehicles in their
queues while the regional accumulations fall. These exploratory outcomes are
saved in exploratory_traffic.json and second_exploration.json.

Before certificate checks, select the following prescribed plans. All ring
inputs are 1 throughout. All entry inputs are 0 over minutes 0--5. Afterwards,
the backup uses entries (0.55,0.50,0.50). A proposal to prioritize Region 1
reduces only its entry input to 0.495. This proposal passes the nominal queue
balance but may overflow under arrival error. These are prescribed plans,
not newly optimized MPC outputs. Their origin does not change the checks.

Test the backup first, then seek a repaired plan along the 0.05 interpolation
grid. The exact queue balance rules out weights below 0.45 for regional error
75 vehicles, queue error 10 vehicles and arrival error +300 veh/h per queue:
Queue 1 at minute 60 is at least 7710 - 11000*(0.495+0.055*lambda).
This is above 2000 at lambda=0.40. The first queue-admissible grid weight is
therefore 0.45. No regional or full-plan feasibility is inferred from this
queue-only calculation; it must be established by the complete certificate.
