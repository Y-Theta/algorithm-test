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
            clas.MaxIncreasingSubarrays_3350(new List<int> { 8, -4, -1, 16, 20 });
        }
    }
}
