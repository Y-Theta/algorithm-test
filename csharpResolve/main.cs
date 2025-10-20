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
            string solution = "1625";
            typeof(ResolveClass1).GetMethod($"Solution_{solution}")
                .Invoke(clas, new object[] { "863376891476", 4, 9 });
            int s = clas.GCD(14,21);
        }
    }
}
