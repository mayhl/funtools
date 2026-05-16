# Destination folder to run simulation and store output
DES_ROOT=$WORKDDIR

# Input file(s) to copy to destination folder
{% if files | length == 1 -%}
INPUT_FILES=("{{ files[0] }}")
{% elif files | length >1 -%}
INPUT_FILES=(
  {%- for file in files %}
  "{{ file }}"
  {%- endfor %}
)
{%- endif %}

{% if folder_type is defined -%}
{% if folder_type == 1 %}
# Attaching job ID suffix to output folder
DES_ROOT="${DES_ROOT}_{{ hpc['job_id']}}"
{% elif folder_type == 2 -%}
# Attaching HPC abbreviate and job ID suffix to output folder
DES_ROOT="${DES_ROOT}_{{ hpc['job_id']}}"
{%- endif %}
{%- endif %}

# Copying input file(s) keeping relative paths
mkdir -pf ${SIM_DPATH}
cd {{ hpc['work_dir'] }} || exit 1
cp --parents -u ${INPUT_FILES} ${DES_ROOT}/

# Executing FUNWAVE
cd ${DES_ROOT} || exit
${EXEC_CMD} {{ nproc }} ./${EXEC_NAME}
