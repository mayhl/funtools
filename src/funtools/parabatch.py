

import param
import pandas as pd
from pathlib import Path
import numpy as np
class Generator(param.Parameterized):

    is_collapsed = param.Boolean(default=False)
    collapse_level = param.Integer(default=0, bounds=(0,None))

    def __init__(self, parameters = None, **kwargs):
        super().__init__(**kwargs)

        self._param_spaces = {}
        self._param_fmts = {}
        self._param_order = []

        if parameters is None: return 
        for p in parameters: self.add(*p)

    def add(self, name, values, fmt=None):

        assert not name in self._param_order, "Parameter '%s' already in list." % name
        self._param_order.append(name)
        self._param_spaces[name] = values

        if fmt is None:
            fmt = lambda i, v, n: ("%%0%dd" % int(np.floor(np.log10(n))+1)) % i
        self._param_fmts[name] = fmt

    def _parse_param(name, data, fmt=None):

        if fmt is None:
            fmt = lambda i, v, n: ("%%0%dd" % int(np.floor(np.log10(n))+1)) % i
        return name, data, fmt

    def _unwrap(self, plist):

        name, data, fmt = plist[0]
        n = len(data)
        cur_p = [(fmt(i, v, n), v) for i, v in enumerate(data)]

        if len(plist) == 1: return [(x,) for x in cur_p]
        
        return [(x,) + y for x in cur_p for y in self._unwrap(plist[1:])]

    def _parse_raw_args(self, args):

        names, values = zip(*args)
        n = self.collapse_level if self.is_collapsed else len(args) 

        dpath = Path(*names[:n])
        dname = ''.join(names[n:])
        return (str(dpath / dname),) + values
   
    def generate(self, order=None):

        if order is None: order = self._param_order
        plist = [(k, self._param_spaces[k], self._param_fmts[k]) for k in order]

        raw_data = [self._parse_raw_args(args) for args in self._unwrap(plist)]
        order.insert(0, 'fpath')
        return LambdaTable(df=pd.DataFrame(raw_data, columns=order))
    


class LambdaTable(param.Parameterized):

    df = param.ClassSelector(default=None, class_= pd.DataFrame)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._steps = [] 
        self._current_step = 0

    @property
    def numbers_steps(self):
        return len(self._steps)

    def apply(self, title, func, *extra_args, **extra_kwargs): 
        
        # PUSH CHECKS INTO VALIDATION STEP?
        # if np.any([not k in cols for k in ikeys]):
        #     raise Exception()
        # if np.any([k in cols for k in okeys]):
        #     raise Exception()
        
   
        self._steps.append(Step(
            title = title,
            step_num = self.numbers_steps,
            func_cls = func,
            exargs = list(extra_args),
            exkwargs = extra_kwargs, 
        ))


    def _execute_step(self, step, df):
        ikeys = step.input_keys
        irows = df[ikeys].iterrows()
        list_kwargs = [(i, {k: r[k] for k in ikeys})  for i, r in irows]
        idxs, data = step.execute(list_kwargs)
        df_update = pd.DataFrame(data, index=idxs, columns=step.output_keys)
        return df_update       


    def _update_dfs(self, df, df_update, valid_idxs):

        kwargs = dict(
            how='outer',
            left_index=True, 
            right_index=True,
            validate='one_to_one',
        )

        update = lambda df: df.merge(df_update, **kwargs)
    
        # Removing failed executes 
        df = update(df[df.index.isin(valid_idxs)])
        self.df = update(self.df)

        return df 


    def execute(self):

        df = self.df

        for i, s in enumerate(self._steps):
            df = self._update_dfs(df, * s.execute(df))
            if len(df) < 1: break
        



class Step(param.Parameterized):

    title     = param.String(default='None', constant=True)
    step_num = param.Integer(default=0, bounds=(0, None), constant=True)
    func_cls = param.Callable()
    exargs   = param.List([], constant=True) 
    exkwargs   = param.Dict({}, constant=True)
    
    
    def _execute(self, idx, **kwargs):

        soft_error = lambda msg: (False, idx, msg)

        #try
        with param.exceptions_summarized():
            func = self.func_cls(**kwargs)

        #except Exception as e:
        #    return soft_error(e)
        
        try:
            rtn_val = func.execute(*self.exargs, **self.exkwargs)
        except Exception as e:
            raise e
            return soft_error(e)
        
        msg = "Derived class '%s' does not return a tuple of size 2." % self.__class__
        if not isinstance(rtn_val, tuple):
            return soft_error(msg)
        elif not len(rtn_val) == 2:
            return soft_error(msg)
        
        is_success, rtn_val = rtn_val
        # NOTE: Some extra tuple layer somewhere 
        return is_success, idx, rtn_val[0]



    def execute(self, df):

        ikeys = self.func_cls.in_keys()
        idx_kwargs = [(i, {k: r[k] for k in ikeys})  for i, r in df[ikeys].iterrows()]
        idx_rtnvals = [self._execute(idx, **kwargs) for idx, kwargs in idx_kwargs]


        okeys = self.func_cls.out_keys()
        err_data = tuple([None]*len(okeys))
        fix_err = lambda is_true, i, r:  (i, r) if is_true else (i, err_data)
        valid_ids = [i for is_true, i, _ in idx_rtnvals if is_true]
        idxs, rtnvals = zip(*[fix_err(*rv) for rv in idx_rtnvals]) 

        return pd.DataFrame(rtnvals, index=idxs, columns=okeys), valid_ids




class LambdaTableFunction(param.Parameterized):

    @classmethod 
    def in_keys(cls):
        return [x for x in cls.param if not x == 'name']
    
    def execute(self):
        return Exception("Derived class '%s' has not implemented _execute." %  self.__class__)

    @classmethod
    def out_keys(self):
        return Exception("Derived class '%s' has not implemented out_keys." %  self.__class__)

    def err_return(self, msg):
        return False, msg
    
    def return_data(self, *args):
        return bool(np.round(np.random.rand())), args