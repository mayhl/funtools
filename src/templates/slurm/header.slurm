#!/bin/bash
## ------------------------
## Required Slurm Directive
## ------------------------
#SLURM -S /bin/bash
#SLURM -A {{ account }}
#SLURM -q {{ queue }}
#SLURM --job-name {{ name }}
#SLURM -N {{ nodes }}
#SLURM -n {{ nprocs }}
#SLURM -t walltime={{ walltime }}
{% if email is defined -%}
#SLURM {{ email }}
{% endif -%}
{% if array_range is defined -%}
#SLURM -J {{ array_range }}
#SLURM -r y
{%- endif %}
