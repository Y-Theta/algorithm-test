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
            var s = clas.BuildTree_0105(new int[] { 3, 9, 20, 15, 7 }, new int[] { 9, 3, 15, 20, 7 });
        }
    }
}
