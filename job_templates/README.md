# Job Templates

This directory contains **template** PBS/SLURM job scripts for HPC clusters. These templates have **all sensitive cluster-specific information removed** (paths, queue names, etc.) and replaced with placeholders.

## 🔒 Security

**Why templates?**
- Actual job files contain cluster-specific paths and queue names
- Exposing these in public repos could reveal infrastructure details
- Templates allow sharing workflow while protecting sensitive info

## 📁 Directory Structure

```
job_templates/          # Template files (tracked in git)
├── README.md          # This file
├── *.template.pbs     # PBS job templates
└── *.template.sh      # Shell script templates

jobs/                   # Your actual job files (gitignored)
├── *.pbs              # Customized for your cluster
└── *.sh               # Not tracked in git
```

## 🚀 Quick Start

### Option 1: Manual Setup

```bash
# 1. Copy template
cp job_templates/run_evaluation_batch.template.pbs jobs/run_evaluation_batch.pbs

# 2. Edit with your cluster info
vim jobs/run_evaluation_batch.pbs

# 3. Replace placeholders:
#   YOUR_QUEUE_NAME → your actual queue (e.g., gpu, debug, batch)
#   /path/to/your/logs/ → your log directory path
#   /path/to/your/project → your project directory path
#   YOUR_NCPUS, YOUR_NGPUS, YOUR_MEMORY, YOUR_WALLTIME

# 4. Submit
qsub jobs/run_evaluation_batch.pbs
```

### Option 2: Batch Replace with sed

```bash
# Copy template
cp job_templates/run_evaluation_batch.template.pbs jobs/run_evaluation_batch.pbs

# Replace all placeholders at once
sed -i 's/YOUR_QUEUE_NAME/gpu/g' jobs/run_evaluation_batch.pbs
sed -i 's|/path/to/your/logs/|/home/myuser/logs/|g' jobs/run_evaluation_batch.pbs
sed -i 's|/path/to/your/project|/home/myuser/project|g' jobs/run_evaluation_batch.pbs
sed -i 's/YOUR_NGPUS/1/g' jobs/run_evaluation_batch.pbs
sed -i 's/YOUR_NCPUS/16/g' jobs/run_evaluation_batch.pbs
sed -i 's/YOUR_MEMORY/64GB/g' jobs/run_evaluation_batch.pbs
sed -i 's/YOUR_WALLTIME/12:00:00/g' jobs/run_evaluation_batch.pbs

# Submit
qsub jobs/run_evaluation_batch.pbs
```

## 📋 Available Templates

### Evaluation Scripts

| Template | Purpose | Default Resources | Time |
|----------|---------|-------------------|------|
| `run_evaluation_test.template.pbs` | Quick test (GPT-2, 100 samples) | 1 GPU, 8 CPUs, 32GB | 1h |
| `run_evaluation_batch.template.pbs` | Full evaluation (all models) | 1 GPU, 16 CPUs, 64GB | 12h |

### Training Scripts

| Template | Purpose | Default Resources | Time |
|----------|---------|-------------------|------|
| `run_cpt.template.pbs` | Continued pretraining (NeMo) | 8 GPUs, 512GB | 12h |
| `run_cpt_test.template.pbs` | Quick CPT test | 1 GPU, 64GB | 2h |

### Data Processing Scripts

| Template | Purpose | Default Resources | Time |
|----------|---------|-------------------|------|
| `preprocess_data.template.pbs` | Data preprocessing | 64 CPUs, 512GB | 8h |
| `preprocess_data_parallel.template.pbs` | Parallel preprocessing | 16 CPUs, 64GB | 4h |
| `build_tokenizer_expansions.template.pbs` | Build tokenizer | 4 CPUs, 32GB | 8h |

### Reference Scripts

| Template | Purpose |
|----------|---------|
| `QUICK_REFERENCE.template.sh` | Common PBS commands cheatsheet |

## 🔧 Customization Guide

### Common Placeholders

Replace these in your customized files:

```bash
# Cluster Configuration
YOUR_QUEUE_NAME          # e.g., gpu, debug, AISG_debug, batch
YOUR_PROJECT_PATH        # e.g., /home/user/project
YOUR_LOG_DIR             # e.g., /home/user/logs

# Resource Requirements
YOUR_NCPUS              # e.g., 4, 8, 16, 64
YOUR_NGPUS              # e.g., 1, 4, 8
YOUR_MEMORY             # e.g., 32GB, 64GB, 512GB
YOUR_WALLTIME           # e.g., 01:00:00, 12:00:00

# Network (for multi-node training)
YOUR_NETWORK_INTERFACE  # e.g., ib0, eth0, eno1

# Container/Environment
YOUR_CONTAINER_PATH     # Path to Enroot/Singularity container (if using)
YOUR_CONDA_ENV         # Conda environment name/path
```

### Example: Complete Setup Script

Save this as `setup_my_jobs.sh`:

```bash
#!/bin/bash
# Customize these for your cluster
QUEUE="gpu"
LOG_DIR="/home/myuser/logs"
PROJECT_DIR="/home/myuser/filipino-morphology-llm"
NETWORK_IF="ib0"

# Create jobs directory
mkdir -p jobs

# Copy and customize all templates
for template in job_templates/*.template.pbs; do
    filename=$(basename "$template" .template.pbs)
    output="jobs/${filename}.pbs"
    
    echo "Creating $output..."
    cp "$template" "$output"
    
    sed -i "s/YOUR_QUEUE_NAME/${QUEUE}/g" "$output"
    sed -i "s|/path/to/your/logs/|${LOG_DIR}/|g" "$output"
    sed -i "s|/path/to/your/project|${PROJECT_DIR}|g" "$output"
    sed -i "s/YOUR_NETWORK_INTERFACE/${NETWORK_IF}/g" "$output"
    
    # Set reasonable defaults for resources
    sed -i "s/YOUR_NGPUS/1/g" "$output"
    sed -i "s/YOUR_NCPUS/16/g" "$output"
    sed -i "s/YOUR_MEMORY/64GB/g" "$output"
    sed -i "s/YOUR_WALLTIME/12:00:00/g" "$output"
done

echo "All job scripts created in jobs/"
echo "Review and adjust resource requirements as needed"
```

Run it:
```bash
bash setup_my_jobs.sh
```

## 🔍 Troubleshooting

### "Queue not found" error
```bash
# Check available queues
qstat -Q        # PBS
sinfo           # SLURM

# Update queue in your job file
vim jobs/your_job.pbs
# Change: #PBS -q YOUR_QUEUE_NAME  →  #PBS -q gpu
```

### "Permission denied" error
```bash
# Ensure log directory exists
mkdir -p /path/to/logs

# Check permissions
ls -la /path/to/logs
chmod 755 /path/to/logs
```

### "Module not found" error
```bash
# Check available modules
module avail

# Update module load commands in script
```

### "Container not found" error
```bash
# Verify container path
ls -la /path/to/container

# For Enroot
enroot list

# For Singularity
singularity exec /path/to/container.sif ls
```

## 📝 Template Guidelines

When creating new templates from your working scripts:

1. ✅ **Remove all absolute paths** specific to your cluster
2. ✅ **Use descriptive placeholders** like `YOUR_QUEUE_NAME`
3. ✅ **Add comments** explaining what needs customization
4. ✅ **Include resource requirements** in comments
5. ✅ **Document dependencies** (containers, modules, etc.)

### Template Header Format

```bash
#!/bin/bash

#PBS -N job_name
#PBS -l select=1:ncpus=YOUR_NCPUS:ngpus=YOUR_NGPUS:mem=YOUR_MEMORY
#PBS -l walltime=YOUR_WALLTIME
#PBS -q YOUR_QUEUE_NAME
#PBS -j oe
#PBS -o /path/to/your/logs/

# ============================================================================
# Job Description - TEMPLATE
# ============================================================================
#
# This is a TEMPLATE file. Copy to jobs/ and customize:
#   1. Replace YOUR_QUEUE_NAME with your cluster queue
#   2. Replace /path/to/your/logs/ with your log directory
#   3. Adjust resources as needed
#
# Usage:
#   cp job_templates/script.template.pbs jobs/script.pbs
#   vim jobs/script.pbs  # Customize
#   qsub jobs/script.pbs
#
# ============================================================================
```

## 🆘 Getting Help

- **Cluster Documentation**: Check your HPC center's docs for PBS/SLURM syntax
- **Resource Limits**: Ask your admin about queue limits and resource availability
- **Project Docs**: See `docs/TRAINING.md` and `docs/EVALUATION.md` for workflow details

## 🔐 Security Reminders

- ❌ **Never commit** files from `jobs/` directory
- ❌ **Never commit** files with real cluster paths
- ✅ **Always use** templates from `job_templates/`
- ✅ **Review diffs** before committing to ensure no leaks

## 📚 Additional Resources

### Common Queue Names by Cluster Type
- **SLURM**: `gpu`, `batch`, `debug`, `normal`, `long`
- **PBS**: `gpu`, `AISG_debug`, `batch`, `normal`
- **LSF**: `gpu`, `normal`, `long`

### Finding Your Cluster Info
```bash
# Queue names
qstat -Q        # PBS
sinfo           # SLURM
bqueues         # LSF

# Node resources
pbsnodes -a     # PBS
sinfo -N -l     # SLURM
bhosts          # LSF

# Your jobs
qstat -u $USER  # PBS
squeue -u $USER # SLURM
bjobs           # LSF

# Network interfaces (for multi-node training)
ip addr show
ifconfig
```

---

**Note**: The `jobs/` directory is gitignored. Your customized scripts stay local and private.
- Adjust resource requirements based on your cluster's availability
