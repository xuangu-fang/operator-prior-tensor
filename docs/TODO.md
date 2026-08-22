# Backlog

Ordered by what the paper needs next, not by what is interesting.

## Deferred: source localization from sparse sensors

Recovering *where* a release happened, rather than what the field is, is the
obvious downstream task and it is harder than it looks.  Reconstruction is
scored against a field we already have; localization needs a dataset built so
that the source is identifiable in the first place -- releases far enough apart
to be distinguishable at the sensor budget, a prior over source location that is
not the one that generated them, and an error measure in metres rather than
NRMSE, which makes it a different experiment rather than another column.

Worth doing as a follow-up branch.  Not worth doing badly to have a downstream
task in the table.

## Not planned

- **More geometric complexity.**  Nineteen layouts across five families is
  already more than the claim needs.  Mazes, office floors and building cores
  were written and reverted: they make the benchmark harder without making the
  argument stronger.
- **Chasing the three-dimensional sensor ceiling.**  The mechanism is understood
  and recorded; further tuning would be effort spent on a limitation rather than
  on the claim.
