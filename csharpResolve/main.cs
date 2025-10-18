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
            string solution = "3397";
            typeof(ResolveClass1).GetMethod($"Solution_{solution}")
                .Invoke(clas, new object[] { new int[] {4 ,4 ,4 ,4 }, 1 });
        }
    }
}
