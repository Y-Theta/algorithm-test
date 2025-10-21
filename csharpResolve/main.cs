using main.csharpResolve;

using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace leetcode
{
    public class Program
    {
        public static void Main(string[] args)
        {
            ResolveClass1 clas = new ResolveClass1();
            string solution = "3346";
            typeof(ResolveClass1).GetMethod($"Solution_{solution}")
                .Invoke(clas, new object[] { new int[] { 23, 54 }, 77, 1 });
            //int s = clas.GCD(14,21);
            //var list = new List<int> { 1, 5, 5, 5, 7, 9, 20, 32, 44 };
            //int aim = 4;
            //int aim1 = 0;
            //Debug.WriteLine(clas.QucikSearchForward(list, aim, 0, list.Count - 1));
            //Debug.WriteLine(clas.QuickSearchBackward(list, aim1, 0, list.Count - 1));
        }
    }
}
