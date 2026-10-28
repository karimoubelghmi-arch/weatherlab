# Public release notes

## Preserved

The statistical engine is copied from the supplied project. Selected Python modules are provided as reference source. These retain original French comments and research notes.

## Added for this portfolio version

An English README, synthetic-data generator and simplified offline Streamlit demo. The example uses a continuous temperature-threshold event rather than reproducing a specific exchange’s bucket-resolution rules.

## Excluded

Original data, wallet identifiers, transaction journals, preferences, state files, caches, logs, node_modules, compiled files, automatic trading runner, signing/execution components and archived agent prototype. Excluding the original datasets avoids republishing third-party data before their redistribution terms are reviewed.

## Limits

This is a curated presentation of the project, not a full replication or a performance audit. Selected reference modules depend on files not included in this subset. Live operation, historical model calibration, order execution and profitability are not validated by the offline demo. AI generated the code implementation; Karim supplied design instructions and manually checked data and statistical outputs.

## Checks performed

Python syntax checked for every included source file. All nine combinations of the demo regression choices and residual-distribution choices returned finite central estimates and positive scale parameters on synthetic data. Streamlit AppTest completed the default demo without exceptions. SciPy emitted numerical overflow warnings while fitting candidate distributions; these checks are not a numerical-stability audit of the original engine. Credential-pattern checks found no matching embedded private keys or API tokens; runtime files and original account histories were excluded.
