Neural Style
===

* dataset
------
    
       coco (just 800 images)
       
* use visdom:
-------    
        python -m visdom.server
        
* train:
-----
        python main.py train --use-gpu --data-root=data --batch-size=2
        
        
* Generate images:
--------------
        python main.py stylize  --model-path='transformer.pth' \
                 --content-path='amber.jpg'\  
                 --result-path='output2.png'\  
                 --use-gpu=False
                 
                 
* result(not so good ,just run 20 epoch and use 800 images) :
----------
![1](https://github.com/ReOneK/Neural-Style/blob/master/ref/1.png)

![2](https://github.com/ReOneK/Neural-Style/blob/master/ref/2.png)
