import pulp

class NeuralNetworkMILP:
    def __init__(self):
        #initialize MILP
        self.prob = pulp.LpProblem("NeuralNetworkMILP", pulp.LpMinimize)
        
        #define variable bounds
        self.x1 = pulp.LpVariable('x1', lowBound=0, upBound=1, cat='Continuous')
        self.x2 = pulp.LpVariable('x2', lowBound=0, upBound=1, cat='Continuous')
        self.x3 = pulp.LpVariable('x3', lowBound=0, upBound=2, cat='Continuous')
        self.x4 = pulp.LpVariable('x4', lowBound=-2, upBound=-1, cat='Continuous')
        self.x5 = pulp.LpVariable('x5', lowBound=-1, upBound=1, cat='Continuous')
        self.x6 = pulp.LpVariable('x6', lowBound=0, upBound=1, cat='Continuous')
        self.x7 = pulp.LpVariable('x7', lowBound=0, upBound=2, cat='Continuous')
        self.x8 = pulp.LpVariable('x8', lowBound=0, upBound=1, cat='Continuous')
        self.x9 = pulp.LpVariable('x9', lowBound=0, upBound=2, cat='Continuous')
        self.x10 = pulp.LpVariable('x10', lowBound=-2.5, upBound=0.5, cat='Continuous')
        
        #binary variables
        self.a7 = pulp.LpVariable('a7', cat='Binary')
        self.a8 = pulp.LpVariable('a8', cat='Binary')
        
        #maximum of b1 and b2
        self.M = 2  
        
        #objective
        self.prob += self.x9 - self.x10, "Minimize_x9_minus_x10"
        
        #affine constraints
        self.prob += self.x3 == self.x1 + self.x2, "Affine_x3"
        self.prob += self.x4 == self.x1 - 2, "Affine_x4"
        self.prob += self.x5 == self.x1 - self.x2, "Affine_x5"
        self.prob += self.x6 == self.x2, "Affine_x6"
        self.prob += self.x9 == self.x7, "Affine_x9"
        self.prob += self.x10 == -self.x7 + self.x8 - 0.5, "Affine_x10"
        
        #max pooling constraints for x7
        self.prob += self.x7 >= self.x3, "Max_x7_geq_x3"
        self.prob += self.x7 >= self.x4, "Max_x7_geq_x4"
        self.prob += self.x7 <= self.x3 + self.M * (1 - self.a7), "Max_x7_leq_x3_plus_Ma7"
        self.prob += self.x7 <= self.x4 + self.M * self.a7, "Max_x7_leq_x4_plus_Ma7"
        
        #max pooling constraints for x8
        self.prob += self.x8 >= self.x5, "Max_x8_geq_x5"
        self.prob += self.x8 >= self.x6, "Max_x8_geq_x6"
        self.prob += self.x8 <= self.x5 + self.M * (1 - self.a8), "Max_x8_leq_x5_plus_Ma8"
        self.prob += self.x8 <= self.x6 + self.M * self.a8, "Max_x8_leq_x6_plus_Ma8"
        
    def solve(self):
        solver = pulp.PULP_CBC_CMD(msg=False)
        self.prob.solve(solver)
        
        if pulp.LpStatus[self.prob.status] != 'Optimal':
            raise Exception("No optimal solution found.")
        
        solution = {
            'x1': pulp.value(self.x1),
            'x2': pulp.value(self.x2),
            'x3': pulp.value(self.x3),
            'x4': pulp.value(self.x4),
            'x5': pulp.value(self.x5),
            'x6': pulp.value(self.x6),
            'x7': pulp.value(self.x7),
            'x8': pulp.value(self.x8),
            'x9': pulp.value(self.x9),
            'x10': pulp.value(self.x10),
            'a7': pulp.value(self.a7),
            'a8': pulp.value(self.a8),
            'Objective': pulp.value(self.prob.objective)
        }
        
        return solution
