#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import itertools as it
import numpy as np

import sys
sys.path.append(r"C:\Program Files\PNN\MIPCL-PY_64")

import mipcl_py.mipshell.mipshell as mp
from ortools.linear_solver import pywraplp


class Problem:
    def __init__(self, text):
        lines = [tuple(map(int, l.split())) for l in text.split("\n")]
        self.n, self.m = lines[0]
        self.cost = np.array([x[0] for x in lines[1:] if len(x)], dtype=np.int)
        self.sets = tuple(
            frozenset(x[1:])
            for x in lines[1:] if len(x)
        )

    def is_feasible(self, sol):
        """Check if the solution is feasible or not"""
        assert(len(sol) == self.m)
        flg = np.zeros(self.n, dtype=np.bool)
        for i, v in enumerate(sol):
            if v:
                for j in self.sets[i]:
                    flg[j] = True
        return np.all(flg)

    def objective(self, sol):
        """Calculate objective value of sol"""
        obj = 0
        for i, v in enumerate(sol):
            if v:
                obj += self.cost[i]
        return obj


class Reducer:
    """Reduce the size of a problem"""

    def __init__(self, p):
        """
        Initialization

        Parameters
        ----------
        p : Problem

        """
        self.p = p
        self._create_reduced_problem()

    def _create_reduced_problem(self):
        self.fixed = {}
        self.sets = {
            i: set(s)
            for i, s in enumerate(self.p.sets)
        }
        self.e2s = defaultdict(set)
        for i in self.sets:
            for x in self.sets[i]:
                self.e2s[x].add(i)

        modified = True
        while modified:
            modified = False
            modified = modified or self._rule2()
            modified = modified or self._rule3()
            modified = modified or self._rule4p()
        assert len(self.fixed) == self.p.m - len(self.sets)

        self.use_cols = tuple(sorted(self.sets.keys()))
        self.use_rows = tuple(sorted(self.e2s.keys()))
        r2i = {x: i for i, x in enumerate(self.use_rows)}
        text = "%d %d\n" % (len(self.use_rows), len(self.use_cols))
        text += "\n".join(
            "%d " % self.p.cost[i] + " ".join(
                str(r2i[x]) for x in self.sets[i]
            )
            for i in self.use_cols
        ) + "\n"
        self.reduced_p = Problem(text)

    def _rule2(self):
        """Remove necessary columns(sets)"""
        modified = False
        xs = tuple(self.e2s.keys())
        for x in xs:
            if x not in self.e2s:
                continue
            if len(self.e2s[x]) == 1:
                i = self.e2s[x][0]
                
                self.fixed[i] = 1
                for y in self.sets[i]:
                    del self.e2s[y]
                del self.sets[i]
                modified = True
        return modified

    def _rule3(self):
        """Remove unnecessary rows(items)"""
        xs = tuple(self.e2s.keys())
        modified = False
        for x, y in it.combinations(xs, 2):
            if x not in self.e2s or y not in self.e2s:
                continue
            if self.e2s[x] <= self.e2s[y]:
                for i in self.e2s[x]:
                    self.sets[i].remove(x)
                del self.e2s[x]
                modified = True
            elif self.e2s[x] >= self.e2s[y]:
                for i in self.e2s[y]:
                    self.sets[i].remove(y)
                del self.e2s[y]
                modified = True
        return modified

    def _rule4p(self):
        """Remove unnecessary columns(sets)"""
        js = tuple(self.sets.keys())
        modified = False
        for i in js:
            if i not in self.sets:
                continue
            ci = self.p.cost[i]
            other = 0
            for x in self.sets[i]:
                other += min(self.p.cost[j] for j in self.e2s[x])
            if ci > other or (len(self.sets[i]) > 1 and ci >= other):
                del self.sets[i]
                self.fixed[i] = 0
                modified = True
        return modified

    def get_original_solution(self, sol):
        """create solution of original problem"""
        assert len(sol) == len(self.use_cols)
        ret = np.zeros(self.p.m, dtype=np.int)
        for i, v in enumerate(sol):
            ret[self.use_cols[i]] = v
        for i in self.fixed:
            ret[i] = self.fixed[i]
        return ret


def get_dual_solution(p):
    """Solve the dual of linear relaxation problem and return solution"""
    solver = pywraplp.Solver('dual', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    u = [
       solver.NumVar(0, solver.infinity(), "u%d" % x) for x in range(p.n)
    ]
    for i in range(p.m):
        solver.Add(sum(u[x] for x in p.sets[i]) <= float(p.cost[i]))
    solver.Maximize(sum(u))
    status = solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL
    return [ux.solution_value() for ux in u]


def get_relative_cost(p, dsol):
    """
    Calculate relative cost

    Parameters
    ----------
    p : Problem
    dsol : np.ndarray of float
        solution of the dual of linear relaxation problem

    Returns
    -------
    rcost : np.array of float
        relative cost

    """
    rcost = np.array(p.cost, dtype=np.float)
    for i in range(p.m):
        for x in p.sets[i]:
            rcost[i] -= dsol[x]
    return rcost


def solve_by_mip(p, time_limit=600, silent=True, bind=None):
    """Solve by MIPCL"""
    mpprob = mp.Problem("setcover")
    x = [mp.Var("x(%d)" % i, type=mp.BIN) for i in range(p.m)]
    mp.minimize(mp.sum_([int(p.cost[i]) * x[i] for i in range(p.m)]))
    for i in range(p.n):
        mp.sum_([x[j] for j in range(p.m) if i in p.sets[j]]) >= 1
    if bind is not None:
        for i in bind:
            x[i] == bind[i]
    mp.optimize(silent=silent, timeLimit=time_limit)
    sol = np.zeros(p.m, dtype=np.int)
    for i in range(p.m):
        if x[i].val > 0.5:
            sol[i] = 1
    assert p.is_feasible(sol)
    return sol


def solve_it(input_data):
    # parse input_data and create problem
    p_org = Problem(input_data)

    # Reduce the size of the problem
    red = Reducer(p_org)
    p = red.reduced_p

    # Calculate a solution of the linear relaxation problem
    # and calculate relative cost
    dsol = get_dual_solution(p)
    rcost = get_relative_cost(p, dsol)

    # Fix some columns(sets) to use
    e2s = defaultdict(list)
    for i in range(p.m):
        for x in p.sets[i]:
            e2s[x].append(i)
    free_cols = set()
    for x in e2s:
        tmp = sorted(e2s[x], key=lambda i: p.cost[i])
        for i in tmp[:1]:
            free_cols.add(i)
    for i in range(p.m):
        if rcost[i] <= 0.01:
            free_cols.add(i)

    # the sets in bind are not used
    bind = {i: 0 for i in range(p.m) if i not in free_cols}

    # solve by mipcl
    sol = solve_by_mip(p, bind=bind)

    # convert to solution of original problem
    sol_org = red.get_original_solution(sol)
    assert(p_org.is_feasible(sol_org))
    obj = p_org.objective(sol_org)

    # make output text
    output_data = "%d %d\n" % (obj, 0)
    output_data += " ".join(map(str, sol_org))
    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/sc_6_1)')

