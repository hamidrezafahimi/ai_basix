

Inside the environment in which you do your ANN stuff, simply run:

```
pip install gym==0.17.3
```

*NOTE ABOUT THE VERSION:* Having python 3.8.10, the gui for the gym environments would not open up. Thus, I tried the above version thanks to [this link](https://stackoverflow.com/questions/73667333/open-ai-gym-environments-dont-render-dont-show-at-all), and it worked!

It seems there is also another dependency for gym's graphic renderings: `pygame`. So do this:

```
sudo add-apt-repository ppa:thopiekar/pygame

sudo apt-get update

sudo apt-get install python3-pygame

pip3 install pgzero

pip install Box2D
```
