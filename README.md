# Efficient Processing of Deep Neural Networks: A Tutorial and Survey 的閱讀

## 前言

```txt
Deep neural networks (DNNs) are currently widely used for many artificial intelligence (AI) applications including computer vision, speech recognition, and robotics. While DNNs deliver state-of-the-art accuracy on many AI tasks, it comes at the cost of high computational complexity. Accordingly, techniques that enable efficient processing of DNNs to improve energy efficiency and throughput without sacrificing application accuracy or increasing hardware cost are critical to the wide deployment of DNNs in AI systems

深度神經網絡（DNN）目前廣泛應用於許多人工智慧（AI）應用，包括計算機視覺、語音識別和機器人技術。雖然DNN在許多AI任務上提供了最先進的準確性，但這是以高計算複雜性為代價的。因此，實現DNN高效處理的技術對於提高能源效率和吞吐量而不犧牲應用準確性或增加硬件成本至關重要，這對於DNN在AI系統中的廣泛部署至關重要。

This article aims to provide a comprehensive tutorial and survey about the recent advances toward the goal of enabling efficient processing of DNNs. Specifically, it will provide an overview of DNNs, discuss various hardware platforms and architectures that support DNNs, and highlight key trends in reducing the computation cost of DNNs either solely via hardware design changes or via joint hardware design and DNN algorithm changes. It will also summarize various development resources that enable researchers and 
practitioners to quickly get started in this field, and highlight important benchmarking metrics and design considerations that should be used for evaluating the rapidly growing number 
of DNN hardware designs, optionally including algorithmic codesigns, being proposed in academia and industry. The reader will take away the following concepts from this article: 
understand the key design considerations for DNNs; be able to evaluate different DNN hardware implementations with benchmarks and comparison metrics; understand the tradeoffs 
between various hardware architectures and platforms; be able to evaluate the utility of various DNN design techniques for efficient processing; and understand recent implementation trends and opportunities.

這篇文章旨在提供關於實現高效深度神經網絡（DNN）處理的最新進展的綜合教程和調查。具體來說，它將概述DNN，討論支持DNN的各種硬件平台和架構，並重點介紹通過硬件設計變更或硬件設計與DNN算法變更相結合來降低DNN計算成本的關鍵趨勢。它還將總結各種開發資源，使研究人員和從業者能夠快速入門這一領域，並強調應用於評估學術界和工業界提出的快速增長的DNN硬件設計（可選包括算法協同設計）的重要基準測試指標和設計考慮因素。讀者將從這篇文章中了解以下概念：

- 理解深度神經網絡（DNN）的關鍵設計考慮因素；

- 能夠使用基準測試和比較指標評估不同的DNN硬件實現；

- 理解不同硬件架構和平台之間的權衡；

- 能夠評估各種DNN設計技術的實用性，以實現高效處理；

- 了解最近的實施趨勢和機會。

KEYWORDS
ASIC; 應用專用集成電路
computer architecture; 計算機架構
convolutional neural networks; 卷積神經網絡
dataflow processing; 數據流處理
deep learning; 深度學習
deep neural networks; 深度神經網絡
energy-efficient accelerators; 能效加速器
low power; 低功耗
machine learning; 機器學習
spatial architectures; 空間架構
VLSI; 超大規模集成電路
```

## 第一節

```txt
Deep neural networks (DNNs) are currently the foundation for many modern artificial intelligence (AI) applications [1]. Since the breakthrough application of DNNs to speech recognition [2] and image recognition [3], the number of applications that use DNNs has exploded. These DNNs are employed in a myriad of applications from selfdriving cars [4], to detecting cancer [5] to playing complex games [6]. In many of these domains, DNNs are now able to exceed human accuracy. The superior performance of DNNs comes from its ability to extract high-level features from raw sensory data after using statistical learning over a large amount of data to obtain an effective representation of an input space. This is different from earlier approaches that use hand-crafted features or rules designed by experts. The superior accuracy of DNNs, however, comes at the cost of high computational complexity. While general-purpose compute engines, especially graphics processing units (GPUs), have been the mainstay for much DNN processing, increasingly there is interest in providing more specialized acceleration of the DNN computation. This article aims to provide an overview of DNNs, the various tools for understanding their behavior, and the techniques being explored to efficiently accelerate their computation.

深度神經網絡（DNN）目前是許多現代人工智慧（AI）應用的基礎[1]。自從DNN在語音識別[2]和圖像識別[3]方面取得突破性應用以來，使用DNN的應用數量激增。這些DNN被應用於從自駕車[4]、癌症檢測[5]到玩複雜遊戲[6]的各種應用中。在許多這些領域，DNN現在能夠超越人類的準確性。DNN的卓越性能來自於其能夠從原始感官數據中提取高級特徵，並通過對大量數據進行統計學習來獲得有效的輸入空間表示。這與早期使用專家設計的手工特徵或規則的方法不同。
然而，DNN的卓越準確性是以高計算複雜性為代價的。雖然通用計算引擎，特別是圖形處理單元（GPU），一直是許多DNN處理的主力，但越來越多的人對提供更專門的DNN計算加速感興趣。這篇文章旨在概述DNN、理解其行為的各種工具以及探索高效加速其計算的技術。

This paper is organized as follows.
•  Section II provides background on the context of why
DNNs are important, their history and applications.
•  Section III gives an overview of the basic components
of DNNs and popular DNN models currently in use.
•  Section IV describes the various resources used for
DNN research and development.
•  Section V describes the various hardware platform
used to process DNNs and the various optimizations 
used to improve throughput and energy efficiency 
without impacting application accuracy (i.e., produce 
bitwise identical results).
•  Section VI discusses how mixed-signal circuits and new
memory technologies can be used for near-data processing to address the expensive data movement that dominates throughput and energy consumption of DNNs.
•  Section VII describes various joint algorithm and hardware optimizations that can be performed on DNNs t
improve both throughput and energy efficiency while 
trying to minimize impact on accuracy.
•  Section VIII describes the key metrics that should be
considered when comparing various DNN designs.

這篇文章的組織如下：

- 第二節 提供了為什麼深度神經網絡（DNN）重要的背景、歷史和應用。

- 第三節 概述了DNN的基本組成部分和目前使用的流行DNN模型。

- 第四節 描述了用於DNN研究和開發的各種資源。

- 第五節 描述了用於處理DNN的各種硬件平台和各種優化技術，以在不影響應用準確性的情況下提高吞吐量和能源效率（即產生位元相同的結果）。

- 第六節 討論了如何使用混合信號電路和新型存儲技術進行近數據處理，以解決主導DNN吞吐量和能耗的昂貴數據移動問題。

- 第七節 描述了可以在DNN上執行的各種聯合算法和硬件優化，以在盡量減少對準確性影響的情況下提高吞吐量和能源效率。

- 第八節 描述了在比較各種DNN設計時應考慮的關鍵指標。
```

## 第二節 介紹 DNN 的背景

A. AI 跟 深度神經網絡(DNNs)

```txt
就是科學家先嘗試用人類的大腦當作機器學習的參考來源

把我們的大腦的運作概念拿出來 我們的大腦主要工作原理如下

大腦的主要計算單元是神經元（neuron）。在一個平均人類大腦中，約有 860 億個神經元。神經元彼此通過樹突（dendrites）和軸突（axon）相連，前者為輸入元素，後者為輸出元素，如圖 2 所示。神經元通過樹突接收輸入信號，對這些信號進行計算，並在軸突上生成輸出信號。這些輸入和輸出信號被稱為激活（activations）。一個神經元的軸突分支與許多其他神經元的樹突相連。軸突分支與樹突之間的連接被稱為突觸（synapse）。

延伸下來到機器學習上

突觸的一個關鍵特性是它能夠對穿過它的信號（𝑥𝑖）進行縮放。這個縮放因子被稱為權重（𝑤𝑖​）。
據推測，大腦的學習方式是通過調整與突觸相關的權重實現的。因此，不同的權重會對相同的輸入產生不同的反應。需要注意的是，學習是指權重因學習刺激而進行的調整，而大腦的組織結構（可以視為類似於程序的部分）並不改變。這一特性使大腦成為機器學習風格算法的極佳靈感來源。

在受大腦啟發的計算範疇內，有一個子領域稱為「脈衝計算」（spiking computing）。該領域的靈感來自於這樣的事實：在樹突和軸突之間的通信是脈衝形式的信號，且信息的傳遞並不僅僅取決於脈衝的幅度，而是與脈衝到達的時間有關。神經元中的計算並不僅依賴於單一值，而是依賴於脈衝的寬度及不同脈衝之間的時間關係。一個受脈衝大腦特性啟發的典型項目是 IBM 的 TrueNorth。

與脈衝計算相對，另一個受大腦啟發的計算子領域是「神經網絡」（neural networks），而這正是本文的重點所在。
```

B. 神經網絡與深度神經網絡 (DNNs)

```txt
神經網絡的靈感來自於這樣的概念：神經元的計算涉及輸入值的加權和。這些加權和對應於突觸進行的信號縮放以及神經元內部對這些值的結合。此外，神經元的輸出不僅僅是加權和，因為如果僅是這樣，則一系列神經元的計算將僅僅是簡單的線性代數操作。相反，在神經元內部存在一個對組合輸入進行運算的功能操作。

這種操作通常被認為是一個非線性函數，它使得神經元只有在輸入值超過某個閾值時才會生成輸出。因此，通過類比，神經網絡對輸入值的加權和應用了一個非線性函數。在第 III-A1 節中，我們將探討這些非線性函數的具體形式。

![alt text](image.png)

圖a 輸入層的神經元接收一些值，並將這些值傳遞到網絡的中間層神經元，該中間層也經常被稱為「隱藏層」。來自一個或多個隱藏層的加權和最終會傳遞到輸出層，輸出層向用戶呈現網絡的最終輸出。 神經元的輸出通常被稱為激活（activations），而突觸則通常被稱為權重（weights）跟大腦的運作一樣

圖b 神經網絡的領域中，有一個稱為深度學習的分支，其特點是神經網絡包含多於三層的結構，即超過一個隱藏層。
```

C. 推理與訓練

```txt
由於深度神經網絡 (DNNs) 是機器學習算法的一種實例，其基本程序在學習執行指定任務的過程中並不會改變。對於 DNNs 而言，這種學習過程涉及確定網絡中的權重（和偏置）的值，這被稱為「訓練網絡」。

一旦訓練完成，程序就可以使用在訓練過程中確定的權重來計算網絡的輸出並執行其任務。使用這些權重運行程序的過程被稱為「推理」（inference）。

本文將重點放在 DNN 推理的高效處理上，而非訓練，因為 DNN 推理通常是在嵌入式設備（而非雲端）上執行，這些設備的資源有限。
```

推理與訓練的區別

1.推理

- 已訓練好的 DNN 使用固定的權重和偏置進行計算，來執行特定任務，例如圖像分類。
- 推理的目的是根據輸入資料產生輸出結果，通常應用於資源有限的設備（如嵌入式設備）。
- 精度需求較低，可採用一些降低精度的技術以提高效率。

2.訓練

- 目的是透過調整權重與偏置，使網絡的輸出更接近目標結果，最小化損失函數 𝐿。
- 利用優化方法（如梯度下降）來逐步更新權重。
- 訓練過程需要大量計算資源與較高的數值精度，並且需要存儲中間輸出以進行反向傳播（Backpropagation）。

訓練的技巧與類型

1.訓練技術

- 使用「批量」損失更新權重：累積多組數據後一次更新，可加速和穩定訓練過程。
- 反向傳播：通過鏈式法則計算每個權重對損失的偏導數，進行高效梯度計算。

2.學習類型

- 監督學習：使用帶有標籤的數據進行訓練（最常見）。
- 無監督學習：用未標籤數據找出結構或群集。
- 半監督學習：結合少量標籤數據和大量未標籤數據。
- 強化學習：基於環境反饋進行動作選擇，目標是最大化長期回報。

3.微調（Fine-Tuning）

- 使用先前訓練好的模型權重作為起點，再針對新數據或約束條件進行調整。
- 優點：加快訓練速度，並可能提升準確性（如遷移學習）。

4.訓練與推理的資源需求差異

- 訓練 需要更多計算資源與存儲，因為必須保存中間結果並進行精確的梯度計算。
- 推理 資源需求較低，特別是在應用於嵌入式設備時，可以採用降低精度的技術以提高運算效率。

深度學習成功的三大因素
1.大量的訓練數據

- 強大的表示學習需要大量數據支持。
- 例子：
  - Facebook 每日接收 10 億張圖片。
  -Walmart 每小時創建 2.5 PB 的客戶數據。
  -YouTube 每分鐘上傳 300 小時影片。
- 巨量數據成為訓練演算法的基石。

2.強大的計算能力

- 半導體技術與計算架構的進步提升了運算效能。
- DNN 中的大量加權求和運算如今能在合理時間內完成，成為訓練與推理的基礎。

3.演算法與工具的進化

- DNN 的成功激發了更多演算法的發展。
- 開源框架（如 TensorFlow、PyTorch 等）的發展使研究者更容易探索與應用 DNN。
- 新技術不僅提升了應用的準確性，也擴展了 DNN 的適用範圍。
