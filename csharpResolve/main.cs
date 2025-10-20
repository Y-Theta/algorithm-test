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
            string solution = "88";
            typeof(ResolveClass1).GetMethod($"Solution_{solution}")
                .Invoke(clas, new object[] { new int[] { 1, 0 }, 1, new int[] { 2 }, 1 });
            int s = clas.GCD(14,21);
        }
    }
}
