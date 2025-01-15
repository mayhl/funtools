
import param 
from pathlib import Path
from string import Template
from jinja2 import Environment, FileSystemLoader



class File(param.Parameterized):

#    _TEMPLATE_FPATH_ = '../templates/hpc_job_submit.pbs'

    _TEMPLATE_DPATH_ = (Path(__file__).parent / '../templates').resolve()
    _TEMPLATE_ = Environment(loader=FileSystemLoader(_TEMPLATE_DPATH_)).get_template('hpc_job_submit.pbs')

    threads_per_node = param.Integer(default=1, bounds=(1,None), constant=True)
    hpc_id = param.String(default=None, constant=True)
    exec_cmd = param.String(default=None, constant=True)
    module_cmds = param.List(default=[], item_type=str, constant=True)
    exe_fpath = param.String(default='funwave-central')

    hpc_work_dpath = param.String(default="", constant=True)
    hpc_home_dpath = param.String(default="", constant=True)

    optimal_subgrid_size = param.Integer(default=200, bounds=(100, None))

    output_mode = param.Selector(default='home', objects=['home', 'group'])
    home_dir = param.String('$HOME')
    work_dir = param.String('$WORKDIR', constant=True)

    @property
    def home_dpath(self):
        match self.output_mode:
            case 'home' : return "$HOME"
            case 'group': return self.hpc_home_dpath

    @property
    def work_dpath(self):
        match self.output_mode:
            case 'home': return "$WORKDIR"
            case 'group': return self.hpc_work_dpath

    @property
    def user_group(self):
        match self.output_mode:
            case 'home': return None
            case 'group': return 'funwave'

    @property
    def is_group_run(self):
        return not (self.output_mode == 'home')

    def write_pbs_files(self, fpath, name, nodes, files=[], n_threads=None):

        if n_threads is None: n_threads=nodes*self.threads_per_node
        files.insert(0, 'input.txt')

        pbs_params = dict(
                PBS_NAME         = name,
                PBS_NODES        = nodes,
                PBS_NODE_NPROC   = self.threads_per_node,
                PBS_EXEC_FPATH   = self.exe_fpath,
                PBS_NPROC        = n_threads,
                PBS_FILES        = "'%s'" % ('\n             '.join(files)),
                PBS_INPUT_DPATH  = self.home_dpath,
                PBS_OUTPUT_DPATH = self.work_dpath,
                PBS_HPC_ID       = self.hpc_id,
                PBS_MODULE_CMDS  = "%s" % ("\n".join(self.module_cmds)),
                PBS_EXEC_CMD     = self.exec_cmd,
                user_group       = self.user_group,
                is_group_run     = self.is_group_run,
            )


        
        # with open(self._TEMPLATE_FPATH_, 'r') as f:
        #     src = Template(f.read())
        #     result = src.substitute(pbs_params)
        #     with open(fpath, 'w') as fw: fw.write(result)

        with open(fpath, 'w') as fh: 
            fh.write(self._TEMPLATE_.render(**pbs_params))

class MikeFile(File):
   def __init__(self, **kwargs):
         
         config = dict(
             threads_per_node = 44,
             hpc_id = None,
             exec_cmd = "aprun -n",
             module_cmds = [
                 "module swap PrgEnv-cray PrgEnv-intel"
             ],
             hpc_home_dpath = "/apps/unsupported/funwave/",
             hpc_work_dpath = "/p/work/funwave/",
         )
         super().__init__(**config, **kwargs)
         