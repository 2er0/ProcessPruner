# ProcessPruner
Automated preprocessing of event logs (Process) 

It allows to process the log before further manually analysing a given log.

## Features

This tools uses footprint matrices like in the Alpha algorithm, which are known in the 
process mining community.
This tool and its algorithm uses footprint matrices to find the core process in en event log.

This allows two very basic put important thinks: 

- Filtering
- Highlight

A given event log can be filtered, to reduce the noise in a log for further processing
and analysing. 
The tool can also be seen as an highlighting tool which points on traces/cases, 
which have not the same flow as the main part of the traces in the log.

## Important to know

This tool only works with if the log contains more traces/cases with the real core process.
If the core process does not exist as it should be, then this tool can not work properly.

## How to use

### Requirements

- Python 3.6 or higher
- Numpy
- Pandas
- tqdm

### Start

Run the tool by executing `python process_pruner.py` or 
include the process into another project via `from process_pruner import process_pruner` 

## Video

Demo: [ Process Pruner ]( https://2er0.github.io/ProcessPruner/ICPM-2019-Process-Pruner.webm )

GitHub Page: [ Web view ]( https://2er0.github.io/ProcessPruner/ )
