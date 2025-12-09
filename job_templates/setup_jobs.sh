#!/bin/bash

# ============================================================================
# Job Setup Script - Create Job Scripts from Templates
# ============================================================================
#
# This script helps you create customized job scripts from templates.
# It will prompt for your cluster configuration and create job files
# in the jobs/ directory with your settings.
#
# Usage:
#   bash job_templates/setup_jobs.sh
#
# Or run non-interactively with environment variables:
#   QUEUE=gpu LOG_DIR=/home/user/logs bash job_templates/setup_jobs.sh
#
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Job Setup Wizard${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "This wizard will help you create customized job scripts from templates."
echo ""

# ============================================================================
# Gather Configuration
# ============================================================================

# Queue name
if [ -z "${QUEUE:-}" ]; then
    echo -e "${YELLOW}Enter your cluster queue name:${NC}"
    echo "  Examples: gpu, debug, batch, AISG_debug"
    echo -n "Queue: "
    read QUEUE
fi
echo -e "${GREEN}✓${NC} Queue: ${QUEUE}"

# Log directory
if [ -z "${LOG_DIR:-}" ]; then
    echo ""
    echo -e "${YELLOW}Enter your log directory (absolute path):${NC}"
    echo "  Example: /home/myuser/logs"
    echo -n "Log directory: "
    read LOG_DIR
fi

# Expand tilde if present
LOG_DIR="${LOG_DIR/#\~/$HOME}"
echo -e "${GREEN}✓${NC} Log directory: ${LOG_DIR}"

# Project directory
if [ -z "${PROJECT_DIR:-}" ]; then
    echo ""
    echo -e "${YELLOW}Enter your project directory (absolute path):${NC}"
    echo "  Current directory: $(pwd)"
    echo -n "Project directory [$(pwd)]: "
    read PROJECT_DIR_INPUT
    PROJECT_DIR="${PROJECT_DIR_INPUT:-$(pwd)}"
fi

# Expand tilde if present
PROJECT_DIR="${PROJECT_DIR/#\~/$HOME}"
echo -e "${GREEN}✓${NC} Project directory: ${PROJECT_DIR}"

# Network interface (for multi-node training)
if [ -z "${NETWORK_IF:-}" ]; then
    echo ""
    echo -e "${YELLOW}Enter your network interface for multi-node training:${NC}"
    echo "  Common options: ib0 (InfiniBand), eth0, eno1"
    echo "  (Check with: ip addr show)"
    echo -n "Network interface [ib0]: "
    read NETWORK_IF_INPUT
    NETWORK_IF="${NETWORK_IF_INPUT:-ib0}"
fi
echo -e "${GREEN}✓${NC} Network interface: ${NETWORK_IF}"

# Default resources
echo ""
echo -e "${YELLOW}Default resource settings:${NC}"
echo "  (You can adjust these in individual job files later)"
echo ""

if [ -z "${DEFAULT_NGPUS:-}" ]; then
    echo -n "Default GPUs [1]: "
    read DEFAULT_NGPUS_INPUT
    DEFAULT_NGPUS="${DEFAULT_NGPUS_INPUT:-1}"
fi
echo -e "${GREEN}✓${NC} Default GPUs: ${DEFAULT_NGPUS}"

if [ -z "${DEFAULT_NCPUS:-}" ]; then
    echo -n "Default CPUs [16]: "
    read DEFAULT_NCPUS_INPUT
    DEFAULT_NCPUS="${DEFAULT_NCPUS_INPUT:-16}"
fi
echo -e "${GREEN}✓${NC} Default CPUs: ${DEFAULT_NCPUS}"

if [ -z "${DEFAULT_MEMORY:-}" ]; then
    echo -n "Default memory [64GB]: "
    read DEFAULT_MEMORY_INPUT
    DEFAULT_MEMORY="${DEFAULT_MEMORY_INPUT:-64GB}"
fi
echo -e "${GREEN}✓${NC} Default memory: ${DEFAULT_MEMORY}"

if [ -z "${DEFAULT_WALLTIME:-}" ]; then
    echo -n "Default walltime [12:00:00]: "
    read DEFAULT_WALLTIME_INPUT
    DEFAULT_WALLTIME="${DEFAULT_WALLTIME_INPUT:-12:00:00}"
fi
echo -e "${GREEN}✓${NC} Default walltime: ${DEFAULT_WALLTIME}"

# ============================================================================
# Create Jobs Directory
# ============================================================================

echo ""
echo -e "${BLUE}Creating jobs directory...${NC}"
mkdir -p jobs

# Create log directory
mkdir -p "${LOG_DIR}"
echo -e "${GREEN}✓${NC} Created log directory: ${LOG_DIR}"

# ============================================================================
# Process Templates
# ============================================================================

echo ""
echo -e "${BLUE}Processing templates...${NC}"
echo ""

TEMPLATE_DIR="job_templates"
OUTPUT_DIR="jobs"

processed=0
skipped=0

for template in ${TEMPLATE_DIR}/*.template.pbs ${TEMPLATE_DIR}/*.template.sh; do
    # Skip if no templates found
    if [ ! -f "$template" ]; then
        continue
    fi
    
    # Get filename without .template extension
    basename=$(basename "$template")
    if [[ "$basename" == *.template.pbs ]]; then
        output_name="${basename%.template.pbs}.pbs"
    elif [[ "$basename" == *.template.sh ]]; then
        output_name="${basename%.template.sh}.sh"
    else
        continue
    fi
    
    output="${OUTPUT_DIR}/${output_name}"
    
    # Check if file already exists
    if [ -f "$output" ]; then
        echo -e "${YELLOW}⚠${NC}  ${output_name} already exists, skipping"
        ((skipped++))
        continue
    fi
    
    # Copy and customize template
    cp "$template" "$output"
    
    # Replace placeholders
    sed -i "s/YOUR_QUEUE_NAME/${QUEUE}/g" "$output"
    sed -i "s|/path/to/your/logs/|${LOG_DIR}/|g" "$output"
    sed -i "s|/path/to/your/project|${PROJECT_DIR}|g" "$output"
    sed -i "s/YOUR_NETWORK_INTERFACE/${NETWORK_IF}/g" "$output"
    sed -i "s/YOUR_NGPUS/${DEFAULT_NGPUS}/g" "$output"
    sed -i "s/YOUR_NCPUS/${DEFAULT_NCPUS}/g" "$output"
    sed -i "s/YOUR_MEMORY/${DEFAULT_MEMORY}/g" "$output"
    sed -i "s/YOUR_WALLTIME/${DEFAULT_WALLTIME}/g" "$output"
    
    # Make shell scripts executable
    if [[ "$output_name" == *.sh ]]; then
        chmod +x "$output"
    fi
    
    echo -e "${GREEN}✓${NC}  Created ${output_name}"
    ((processed++))
done

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Setup Complete!${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "Processed ${GREEN}${processed}${NC} template(s)"
if [ $skipped -gt 0 ]; then
    echo -e "Skipped ${YELLOW}${skipped}${NC} existing file(s)"
fi
echo ""
echo "Job scripts created in: ${OUTPUT_DIR}/"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review job scripts in jobs/ directory"
echo "  2. Adjust resource requirements if needed"
echo "  3. Submit jobs: qsub jobs/your_job.pbs"
echo ""
echo -e "${YELLOW}Configuration used:${NC}"
echo "  Queue: ${QUEUE}"
echo "  Log directory: ${LOG_DIR}"
echo "  Project directory: ${PROJECT_DIR}"
echo "  Network interface: ${NETWORK_IF}"
echo "  Default resources: ${DEFAULT_NGPUS} GPU(s), ${DEFAULT_NCPUS} CPU(s), ${DEFAULT_MEMORY}, ${DEFAULT_WALLTIME}"
echo ""
echo -e "${GREEN}✓${NC} All set! You can now submit jobs."
echo ""
