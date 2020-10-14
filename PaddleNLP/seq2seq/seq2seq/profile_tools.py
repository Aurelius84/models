# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cProfile, pstats, StringIO
import six
import time
import paddle.fluid.profiler as profiler


class ProfileTimer(object):
    def __init__(self, dump_file=None, sortby='cumulative', use_fluid=True):
        self.dump_file = "./declarative_benchmark/" + dump_file
        self.sortby = sortby
        self._init_env()
        self.use_fluid = use_fluid

    def _init_env(self):
        self.pr = cProfile.Profile()
        self._is_enable = False
        self._is_disable = False
        self._t_start = 0
        self._t_end = 0

    def enable(self):
        if not self._is_enable:
            print("start profiling....")
            if self.use_fluid:
                profiler.start_profiler(state="All", tracer_option='OpDetail')
            else:
                self._t_start = time.time()
                self.pr.enable()

            self._is_enable = True

    def disable(self):
        if not self._is_disable:
            print('end profiling....')
            if self.use_fluid:
                profiler.stop_profiler(
                    sorted_key='total',
                    profile_path=self.dump_file + ".pd.prof")
            else:
                self._t_end = time.time()
                self.pr.disable()

            self._is_disable = True

    def reset(self):
        self._init_env()

    def save(self, dump=True, file_name=None):
        if self.use_fluid:
            return

        if file_name is None:
            file_name = self.dump_file

        assert isinstance(self.dump_file, six.string_types)
        print("saving profile info into {}....".format(file_name))
        self.pr.dump_stats(file_name + '.prof')

        # whether dump into terminal.
        if dump:
            out = self.dump()
            with open(file_name + '.readable', 'w') as f:
                f.writelines("cost time : {} s.\n".format(self._t_end -
                                                          self._t_start))
                f.write(out)

    def dump(self):
        s = StringIO.StringIO()
        ps = pstats.Stats(self.pr, stream=s).sort_stats(self.sortby)
        ps.print_stats()
        out = s.getvalue()
        print(out)
        return out
