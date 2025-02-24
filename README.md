# 基于计算机视觉识别和鼠标代理的自动化接住高速移动物体游戏辅助脚本

## 一、效果预览：

https://github.com/user-attachments/assets/0dc5c4fc-b98d-4136-a14e-c6309ac55de1

## 二、游戏介绍：汤圆大作战

元宵节被我妈指派玩单位的线上平台小游戏，简单来说任务就是刷小游戏排位-获得积分-获得积分赛排名-领取对应等级的奖品。于是就有了这个脚本~

<div align="center">
<img width="200" alt="1739456625(1)" src="https://github.com/user-attachments/assets/1f5b167b-1abc-4ad8-b59c-f12cbc66d10f" />
</div>

**游戏目标：**

初始给定60s，在倒计时未结束前尽可能通过（水平）移动底部的熊猫篮子<img width="60" alt="1739457270(1)" src="https://github.com/user-attachments/assets/b31f1efd-f288-4e44-a4ce-b262567fe1f1" />
接住从屏幕顶端落下道具来获得尽可能多的分数

**道具类型：**

- **汤圆**<img width="30" alt="1739456625(1)" src="https://github.com/user-attachments/assets/48cbb5d8-ee2f-442c-b17b-975cf52fa1bf" />：+10分
- **炸弹**<img width="30" alt="1739456412(1)" src="https://github.com/user-attachments/assets/aae1175c-af5e-474b-9abc-62c03040515f" />：-5分
- **保护罩**<img width="30" alt="1739456442(1)" src="https://github.com/user-attachments/assets/84654b4d-ccac-48d9-b30e-301223bd8056" />：在5s内使炸弹的减分效果失效
- **加时时钟**<img width="30" alt="1739456653(1)" src="https://github.com/user-attachments/assets/30d63485-a2aa-4a45-99f8-cebc4e6b6db2" />：倒计时+5s
- **减时时钟**<img width="30" alt="1739456284" src="https://github.com/user-attachments/assets/3133e4d7-f2d8-48d2-acd0-804c2f173c98" />：倒计时-10s

道具出现的比例大致为5:4:0.6:0.5:1


游戏规则非常简单，实际中能观察到一些其他的潜在规则设置：

- 每个道具的碰撞体积大致就是各自图像的最小外接矩形，底部的篮子的**碰撞体积**不用考虑熊猫，只有篮子那么大，碰撞只需要篮子的碰撞矩形与道具矩形相接即可
- 总共有5个轨道，同一水平线下只会出现一个道具
- 道具下落的速度会随着游戏时长的增加越来越快
- 相邻下落的两个道具之间的距离和间隔都不固定，但应该也有一个均值，并且
  - 在多数时候，篮子可以穿过相邻下落的两个道具之间的间隙而不触碰到任一道具
  - 在后续速度极快时，从体验上看似乎游戏设计者没有刻意保持相邻下落的两个道具之间的距离很近（这意味着要接/躲两个相邻道具之间的反应时间要非常快），速度越来越大时相邻下落的两个道具之间的距离总体也会按一定趋势增加

这个游戏玩多了就意识到了一些**特性**：

- 游戏下落的道具本身带有随机性，有时候游戏非常快就结束并不是因为失误吃到很多减时时钟，而是因为出现的加时时钟太少了，倒计时入不敷出。这种情况平均6局就有5局是这样，只有少数情况下没有失误的情况下可以持续玩下去一直到速度超过正常人的反应力。
- 由于上一条的情况，该游戏中后期最重要的任务反而不是接住所有汤圆，而是冒着吃到炸弹的风险也应该尽可能接住所有加时时钟，因为只要有时间总能慢慢积累分数；相反，一定要尽可能避免吃到减时时钟，特别是连续出现一串减时时钟时，一连串10s的缩减非常致命

## 三、编写程序脚本的动机

我也给其他几个朋友都玩玩过，大学生的极限差不多都在5000左右，运气好一点也最多在6000~7000，但是排行榜上前200名都在8000+！就算单位每个职工的分数都是各自家里的年轻人在发力，这个数值也很奇怪对不对？甚至某天10:30开放游戏，10:37排行榜上就出现了一大批8000+的玩家。要知道玩8000分至少要5分钟，又因为随机性至少要刷三四局才可以做到，还不考虑失误，至此确认这个数据肯定有猫腻。

大学生刷了几天才发现这个情况当然出奇的不爽，凭什么努力玩游戏还不如神秘的力量？于是想到不如我来手动用程序来刷榜，一是这样每天就不用麻烦地在那边刷这个小游戏（虽然后来发现写程序花了一两天更麻烦），二是一口气刷到前几名也特别酷炫。

## 四、程序建模

根据上面提到的潜在规则第2、4条很容易想到脚本的核心模型：

<img width="400" alt="image" src="https://github.com/user-attachments/assets/2b17533d-13cf-4f24-8369-b1a9ab71f309" />

简单来说，只要实时获取屏幕中所有物体的位置，然后basket只要检索当最近的item对应的$`y_{item}<dist`$时，判断item的类型然后执行操作：

- 接住：$`x_{basket}=x_{item}`$
- 躲避：$`x_{basket}=x_{left} or x_{right}`$ ，简单起见就设置为躲到距离该item最远的一个轨道

只要保证实时识别的更新率够快，操作的反应够迅速，按道理就能实现完美操作是不是！但是下面要说的就是一串踩坑经验……

### 4.1 截图

主要是通过USB连接的ADB调试来实现的，然后调用Github库[scrcpy](https://github.com/Genymobile/scrcpy)来实现的快速截图，后续为了加快获取窗口的速度用AI优化了一下程序，基本可以控制在0.001s内刷新。


### 4.2 识别图像

OpenCV的库中已经有很完善的根据模板来识别给定图像中的位置的包了，但是类似截图汤圆<img width="30" alt="1739456625(1)" src="https://github.com/user-attachments/assets/48cbb5d8-ee2f-442c-b17b-975cf52fa1bf" />的图像匹配情况很糟糕，特别容易识别错位置。在这个基础上优化了边框检测、大小调整和灰度检测等等网上博客介绍的方法都没有很好的改善，后来发现<img width="30" alt="1739456625(1)" src="https://github.com/user-attachments/assets/9a87e3fa-7af7-4109-a15e-89ce9f258abd" />去掉截图背景的干扰后效果反而非常好。说明截图背景的颜色可能真的干扰很大？

然后因为要单模板-多匹配的问题，可能会匹配到很多重复的图像位置，手动调试出合适的相似匹配阈值后，这里参考了网上博客就选用了和NMS（极大值胜出算法）来去重。

注：但是网上抠图软件扣去背景后的图像<img width="30" alt="1739456625(1)" src="https://github.com/user-attachments/assets/4187ee48-fe43-48df-b7bc-8aeebe9188e1" />仍然效果很差，猜测抠图时对除背景之外的部分也造成了不容易观察的影响，对利用相关系数匹配的算法干扰很大

### 4.3 鼠标控制

直接使用pyautogui库即可控制电脑鼠标位置进而控制手机上的滑动操作，但要注意默认会保持0.1s延迟保证程序运行要是出错，电脑不会因为超高速的鼠标乱飞而失控。在调试没问题后要把它设为0.0001s左右比较好。

### 4.4 反应距离设置

我们根据好坏将所有道具分为“好东西”和“坏东西”两个集合，然后根据集合来判断“接住”、“躲避”的逻辑。

尽管图像识别会计算所有物体的下沿和basket上沿绝对距离最小的物体信息，basket是根据这个定义下最近的物体来执行动作，但过大的反应距离仍然可能会在某些时候使得basket提前反应，触碰到之前本来要躲避的、已经快要下落出屏幕的上一个坏东西；过小的反应距离理论上只要计算机识别、反应速度够快也完全能接住，所以一开始的设定是偏小的。

到目前为止，脚本已经能顺利跑起来了！

### 4.5 优化

**搜索优化：**

单纯执行以上的过程仍然只能将游戏积累到1000分左右（还记得人工可以实现5000分左右），原因是**识别图像的速度太慢**了。想一下也很好理解，对于每个模板都要在截屏上卷积滚动一遍显然非常耗费时间，一次完整的屏幕识别需要0.1单位的时间，那么到了后期越来越快的时候程序反应不过来再正常不过。

最终我做了如下优化：

- **减少搜索区域**：只在反应距离上方的一块矩形区域搜索，因为屏幕太下方的位置道具已经来不及了，屏幕太上方的道具不会引起basket的判断逻辑
- **提前终止判定**：因为搜索区域较小、同一行只会有一个道具，所以如果按照模板的搜索循环中已经检索到了$`k`$个物体，就提前终止搜索

  通过这个方法，基本上循环可以缩减到0.04s的刷新率，游戏可以基本累积到3000分+

**自适应反应距离：**

为什么程序还没有超过人工？核心的原因是在程序“识别-反应”的逻辑里，所有的操作已经优化的不能再快了（至少我没找到怎么才能更快），当游戏进行到3000分后的速度已经不是程序的刷新率能反应过来的了，于是自然地开始思考反应距离$`dist`$的作用。在现实生活中，快速规避某个障碍物的这一事件最直接对应的就是司机驾驶车辆遇到突发情况的刹车过程，我们往往在高中物理题中就做到关于**反应时间-刹车距离**的计算题，在这个脚本中也许也能控制合适的“反应距离”来为不那么充分快速反应的程序提供能够安全操作的能力。

此前已经提到，**过长的反应距离**带来的表现就是basket过早移动、触碰到上个本该躲避的物体，**过短的反应距离**表现为basket用篮子的侧边而不是上沿“撞击”道具的侧边来接到、说明即将错过。但是在不同的游戏时间内，不同的道具下落速度对应了不同的最优反应距离，一个自然的想法是用某种函数来拟合"dits-t"的变化曲线，但在缺乏数据的情况下只能人工掐表进行瞪眼调试，最终采用了**分段线性插值**的方法确定了最高能打到11000的拟合函数，也是最终刷榜到第四的方法。


以下提供一个最终版本的方法示例结果，没有使用最高分的打法过程是因为随机性导致每次进行游戏前并不知道能不能打到最高分，所以一直没有录制到。


<div align="center">
    <a href="https://youtu.be/vt5fpE0bzSY">
        <img src="https://github.com/user-attachments/assets/128c0ecb-b3f1-4e50-9553-23c5cd92bf30" width="700" alt="Watch the video">
    </a>
</div>


### 4.6 未完成的优化展望

前面已经提到理想的反应距离应该是关于下落速度$`v`$和游戏进行时间$`t`$两者或其一的函数$`dist(t,v)`$，直觉上只要知道$`v`$就足够了，但是在online的环境下还不清楚怎么搜集数据来拟合它，这种情况最直接的做法就是不如训练一个强化学习模型来让它自己探索，但因为还不会用DQN、CPU也不知道跑不跑的起来就暂时搁置了。但是直觉上将截屏的图片作为特征，或者进一步提取出所有item的信息来构成训练神经网络的输入信号应该是没问题的，但是真实标签怎么来定还是一个待解决的问题。

## 五、程序运行说明

### 5.1 主要依赖库:

- opencv-python
- numpy
- pygame
- pyautogui
- pywin32
- keyboard


### 5.2 使用方法：

- 下载scrcpy并解压到项目根目录，确保各依赖库下载完毕
- 用USB连接Android手机,开启USB调试
- 运行脚本（按ESC键退出程序）

注：代码中，游戏窗口标题根据我的手机设置为了"JAD-AL00"，不修改时运行脚本能够启动投屏但启动屏幕控制代理会失败，这个时候看一下投屏窗口的名称，修改"JAD-AL00"为该名称即可。
  
### 5.3 调试说明：

- 修改show=True参数可显示实时识别效果
- 调整template_params中的阈值可改变识别灵敏度
- reflect_dist参数控制反应距离
- begin和end参数控制检测范围



