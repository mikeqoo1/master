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

