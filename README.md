# My experience using python (in no perticular order)

Here you will find section by section my experience with python and what I have learned using it.

## 101AlphasImplementation

It is my implementation of famous predictive features from a paper [101 Formulaic Alphas]. I used numpy and numba to vectorize and speed up computation. Use `python version 3.10` here.

If you need more speed use [mutliproccessing with shared objects](#shared-objects)

## modelingTermStructure

Shows a way of modeling futures (financial product) term structure. Every future has its own life time, sometimes years ahead. Often they are traded in the present. [TermStructure] show diviances of price of each future contract from the current contract. Looking at the term structre can give clues about market anticipation - what is priced in already.

Please see it as my exploration, it is by no means a production ready algorithm. It just shows my dirty work while I am experementing.

## solvingPythonBottleneckWithCython

It was my way of resolving a huge performance bottleneck in a part of quite big algorithm. The issue was a **python loop** that itereates over a dataframe and simulates trades under curtain conditions. Native python is **slow**, and victorization was not possible. Computation duration avaraged to rughly 200 years based on a linear estiamtion. I had to find a way to speed it up dramatically. I created a C module at first, but then used cython instead. Because of cythons optimizations it was even faster then my C module, and significantly faster then python native version. **I successfully decreased computation duration from roughly 200 years down to 5 days, resulting in about 7k performance increase.**

### Shared Objects

Python is single threaded by design. It has several implications I wont elaborate here. Python is also dynamically typed, and there are some drawbacks to it, speed is one of them. Python sometimes takes a whole object as an argument and sometimes a reference to it. Hard to control it as a User. And in multiprocessing it just duplicates the object, even if you just want to read it, resulting in bloated memory and eventually a system crash. If you use linux and have a **swap partition** it will dump it to a disc, but it kinda defeats the purpose of a perforamnce optimization. It is obvoisly good not to share an object that you want to write accross async mutlithreaded environments, because of race conditions. That being said, there are ways to avoid duplication of objects such as huge DataFrames with native `multiprocessing.Manager`:

```python
from multiprocessing import Manager

mgr = Manager()
ns = mgr.Namespace()
ns.df = my_dataframe
```

I used it a lot to speed up computation and shared in memory objects across threads.

### Cython Implementation

In order to use the [engine.pyx] you need to compile it, but create conda env first.

```bash
conda env create -f environment.yml
conda activate engine
python setup.py build
```

Then you can work with it by using the function `getTradesCythonEngine(*args)`

I can show endless optimization tricks here but I think you got the idea.

## tradingAlgorithmSnipped (QuantConnect:LeanEngine)

This is a snipped of running in production trading algorithm. The code will not run on your machine, as it misses key ingredients. However, you can see here how I implemented routine jobs using [LEAN Engine] inside the algorithm, by leveraging engines abilities. I can not tell what they do but it is running in production for two years without isseus.

**NOTE** I renamed some functions, because their names were pretty clear about what they do, and I would expose to much. Another thing is that there are a lot of static DUMMY variables in the code, I put them there to reduce my exposure. There is no dead code lying around in production, in this snipped it might be the case.

I used inheritance in [Alphas.py] I usually try to avoid it, but here it made sence, beacuse I used to instantiated all of the classes in a [AlphasBuilder.py] in one go:

```python
class AlphasBuilder:
    def scheduleAlphas(self):
        alphasAdded = {}
        missingAlphas = []
        for alpha in self.alphas:
            instance: AlphaFactor = globals()[alpha](self.Algo)
```

In fact it is not even a builder by [DesignPatterns] definition. It doesnt return `self`. However I wrote this code 3 years ago, didnt know about design aptterns, and it is running in production for two years now. When something works why change it?
Another point is that I used `globals()` which might not be the best solution ever, but the list of classes was quite big and i was lazy to write them all down, as the list also changed frequently at that time. It works, so i didnt bother.

In production algorithm pushes events to my API, see [api.py].

`Trade` in [tradeBook.py] just represents what a trade is, and has some basic functionality. In addition there are some functions which enable communication with the API.

There is much more to complain about, but again, it works, and is used only by me.

I dont abstract to much cuz I really thing it makes the code less readable. Probably if I ever rewrite the algorithm I would use new knoledge and exprience I got, and would use some of the [DesignPatterns], like Builder, StartegyPattern, Iterators and some more. These are my favorites now.

## myAPI

**NOTE** An API example that I use, it is pretty straight forward. I show only some services and endpoints, because they are implemented in similar fasion. It is based on the documentation of fastAPI, very good documentation btw.

First of all it overcomes limitiations of the [LEAN Engine] that I wont elaborate here as it will take to long.That was actually the initial goal why I designed that. But over time it has grown substantially. Now it is pulling a lot of data from many sources, runs some tasks in background and it plays a major role in my setup. Sorry for not showing more, but it doesnt feel good to expose my API that runs in production, I hope you understand.

However it shows how I implemented the pushing and pulling trade information between my algorithm and the Database thorugh the API. I also used here a basic SQLLite database, in production it is more sophisticated than that.

I also show here a basic Dockerfile. In production I use NGINX and Gunicorn setup. NGINX as a reverse proxi.

[101 Formulaic Alphas]:https://arxiv.org/pdf/1601.00991.pdf
[TermStructure]:modelingTermStructure/TermStructure.ipynb
[engine.pyx]: solvingPythonBottleneckWithCython/engine.pyx
[LEAN Engine]: https://github.com/QuantConnect/Lean
[Alphas.py]: tradingAlgorithmSnipped/Alphas.py
[AlphasBuilder.py]: tradingAlgorithmSnipped/AlphasBuilder.py
[api.py]: tradingAlgorithmSnipped/api.py
[tradeBook.py]: tradingAlgorithmSnipped/tradeBook.py
[DesignPatterns]: https://www.amazon.de/Patterns-Elements-Reusable-Object-Oriented-Software/dp/0201633612/ref=asc_df_0201633612/?tag=googshopde-21&linkCode=df0&hvadid=310687606304&hvpos=&hvnetw=g&hvrand=15615559861798841127&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9044083&hvtargid=pla-395340045790&psc=1&th=1&psc=1&tag=&ref=&adgrpid=57334095330&hvpone=&hvptwo=&hvadid=310687606304&hvpos=&hvnetw=g&hvrand=15615559861798841127&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9044083&hvtargid=pla-395340045790