import unittest

import numpy as np
from RandomSeedSearchCV import RandomSeedSearchCV, rand_util

class test_RandomSeedSearchCV(unittest.TestCase):
    def test_example(self):
        self.assertEqual(5,5)

class test_rand_util(unittest.TestCase):
    def test_dtype_float(self,n=20):
        for i in range(n):
            dist = np.random.choice(['uniform','normal'])
            inp1 = np.random.uniform(0,10)
            inp2 = np.random.uniform(0,10)
            r = rand_util(inp1=inp1,inp2=inp2,dist=dist,dtype=float)
            self.assertTrue(isinstance(r,float))
    def test_dtype_int(self,n=20):
        for i in range(n):
            dist = np.random.choice(['uniform','normal'])
            inp1 = np.random.randint(0,10)
            inp2 = np.random.randint(0,10)
            r = rand_util(inp1=inp1,inp2=inp2,dist=dist,dtype=int)
            self.assertTrue(isinstance(r,int))
    def test_P_None_1(self):
        r = rand_util(P_None=1)
        self.assertIsNone(r)
    def test_P_None_True(self):
        r = rand_util(P_None=True)
        self.assertIsNone(r)
    def test_override(self):
        r = rand_util(override=5)
        self.assertEqual(r,5)


if __name__ == '__main__':
    unittest.main()