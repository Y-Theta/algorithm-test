#include "0-common.h"

class Foo {
public:
    Foo() {
        
    }

    static bool firstflag;
    static bool secondflag;

    void first(function<void()> printFirst) {
        
        // printFirst() outputs "first". Do not change or remove this line.
        firstflag= false;
        printFirst();
        firstflag= true;
    }

    void second(function<void()> printSecond) {
        
        // printSecond() outputs "second". Do not change or remove this line.
        secondflag = false;
        while (!firstflag)
        {
            sleep(1000);
        }
        printSecond();
        secondflag = true;
    }

    void third(function<void()> printThird) {
        
        // printThird() outputs "third". Do not change or remove this line.
         while (!secondflag)
        {
            _sleep(1000);
        }
        printThird();
    }
};