# Job Submission Scripts

This directory contains Slurm/PBS job submission scripts for the project.

## Available Jobs

### `submit_cpt_gemma3_1b.sh`
Continued pretraining of Gemma 3 1B on SEA-PILE-v2 dataset.

**Usage:**
```bash
# Edit the script to set your WANDB_API_KEY first!
qsub jobs/submit_cpt_gemma3_1b.sh
```

**Configuration:**
- 1 node, 8 GPUs
- 48 hours walltime
- Hopper partition
- Logs to: `logs/gemma3-1b-cpt.log`

**Before submitting:**
1. Export your `WANDB_API_KEY` with `export WANDB_API_KEY="your wandb key"`
2. Activate the environment (`source path/to/env/bin/activate`)
3. Ensure data is prepared with `python src/data_preprocessing/prepare_seapile.py`

## Monitoring Jobs

```bash
# Check job status
qstat -u $USER

# Check job details
qstat -f <JOB_ID>

# Monitor log in real-time
tail -f logs/gemma3-1b-cpt.log

# Cancel a job
qdel <JOB_ID>
```

## Customizing Jobs

The job scripts call `scripts/run_cpt_gemma3_1b.py` with various arguments.
You can modify the script to change:

- `--max-steps`: Training duration
- `--lr`: Learning rate
- `--global-batch-size`: Batch size
- `--checkpoint-dir`: Where to save checkpoints
- And more...

See `python scripts/run_cpt_gemma3_1b.py --help` for all options.
