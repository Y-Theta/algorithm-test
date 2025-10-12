using main.csharpResolve;

using System;
using System.Collections.Generic;

namespace leetcode
{
    public class Program
    {
        public static void Main(string[] args)
        {
            ResolveClass1 clas = new ResolveClass1();
            List<long> sum = new List<long>();
            clas.InventoryManagement_0159(new int[] { 0, 0, 1, 3, 4, 5, 0, 7, 6, 7 },9);
        }
    }
}
