<h1 align='center'>论文分析</h1>

# 1 短语提取

从论文摘要中提取并过滤关键短语，主要使用基于BERT的KeyBERT模型和spaCy自然语言处理工具。

## 1.1使用到的模型

- **SciBERT模型**: 使用专门针对科学文献训练的BERT变体
- **spaCy英文模型**: 用于词性标注(POS)分析
- **KeyBERT模型**: 基于BERT的关键词提取工具

## 1.2 相关操作

首先读取每个论文的摘要，对摘要进行预处理。将摘要文本转写为小写，并使用SciBERT分割句子得到短语。

spaCy对SciBERT分割出的短语中的词进行词性标注，使用KeyBERT提取2-5实词的关键短语，借助SciBERT对短语进行语义分析，挑选出符合条件的短语

对于单个文献摘要提取出的短语可能具有相似的含义，所以使用Jaccard相似度过滤去除冗余短语，就得到的初步的需要的短语。

## 1.3 优化操作

对于得到的短语，其中有些许部分虽然满足2-5个实词的短语，也通过语义分析挑选出所谓符合条件的短语，但由于未限制短语的数量，可能挑选出的短语不符合本文研究的内容要求，因此需要再人工简单筛选一下，通过对少量结果的查看，挑选出不符合条件的短语，再以此为基础筛选与不符合条件的语句相似的短语，如此反复，从1097篇文章中得到了6434条短语。



# 2 短语分类

上面得到的6434条短语隶属于各个年份不同分领域不同研究方向的内容，现在需要将这些短语进行分类，得到每篇文章在研究什么内容。

## 2.1自底而上探索分类

使用深度学习all-MiniLM-L6-v2模型将所有短语转换为数学向量，方便计算机理解短语之间的语义关系。

采用自底向上的层次聚类方法，将相似的短语分组到一起。当前指定聚类的层次，一层4类，二层5类和三层7类，形成从粗到细的分类体系。

使用scipy.cluster.hierarchy包通过语义相似对短语进行聚类，随后通过deepseek对这些类的内容进行总结，得到每个类的名称。

## 2.2 自顶向下分类

但分类不能完全依赖，它给我们提供了一些分类的方向和例子，再通过人为的修正，得到应该分成的类别，再自顶向下计算短语向量的余弦相似度，最终将这些短语分成了一层7类，二层3类，三层3类

简单展示：

![classification_tree_right](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/classification_tree_right.png)

简单统计每个类别有多少

| 类别                                                        | 数量 |
| ----------------------------------------------------------- | ---- |
| Human–AI Cognition & Decision Mechanisms                    | 29   |
| Trust, Transparency & Explainability                        | 5324 |
| Emotion, Empathy & Social Acceptance                        | 173  |
| Algorithm Perceptions, Attitudes & Psychological Mechanisms | 79   |
| Human–AI Teaming & Control                                  | 589  |
| Human–AI Service & Psychological Mechanisms                 | 186  |
| Human–AI Collaboration, Safety & Societal Governance        | 54   |

![image-20250830180723016](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/image-20250830180723016.png)

| 二级主题                                     | 数值 |
| :------------------------------------------- | :--: |
| Explainability & Transparency                | 3960 |
| Collaboration & Coordination                 |  77  |
| Governance & Accountability                  | 896  |
| Trust Formation & Calibration                | 468  |
| Augmentation & Immersion                     |  89  |
| Empathy & Social Interaction                 |  29  |
| Control & Intervention                       | 462  |
| Role Allocation & Adaptation                 |  50  |
| Psychological Mechanisms & Anthropomorphism  |  73  |
| Emotional Reactions & Mental Workload        | 140  |
| Experience & Interaction                     |  24  |
| Cognitive & Decision Styles                  |  53  |
| Decision Aids & Cognitive Offloading         |  10  |
| Social Signals & Norms                       |  48  |
| Information Processing & Situation Awareness |  18  |
| Algorithmic Attitudes                        |  11  |
| Governance & Ethics                          |  6   |
| Mental Models & Biases                       |  15  |
| Social Acceptance & Resistance               |  4   |
| Automation Reliance & Biases                 |  1   |

![image-20250830180558234](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/image-20250830180558234.png)

论文数量随年份变化的趋势

Trust Transparency Explainability

![Trust_Transparency_Explainability_Trend](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/Trust_Transparency_Explainability_Trend.png)

Human–AI Cognition Decision Mechanisms

![Human–AI_Cognition_Decision_Mechanisms_Trend](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/Human–AI_Cognition_Decision_Mechanisms_Trend.png)

Emotion, Empathy & Social Acceptance

![Emotion_Empathy_Social_Acceptance_Trend](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/Emotion_Empathy_Social_Acceptance_Trend.png)

Algorithm Perceptions, Attitudes & Psychological Mechanisms

![Algorithm_Perceptions_Attitudes_Psychological_Mechanisms_Trend](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/Algorithm_Perceptions_Attitudes_Psychological_Mechanisms_Trend.png)

简单呈现这四个，有需要再画。下面是热力图的呈现方式

![Heatmap_7_Topic1_Over_Years_single](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/Heatmap_7_Topic1_Over_Years_single.png)

![Heatmap_7_Topic1_Over_Years](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/Heatmap_7_Topic1_Over_Years.png)

# 53个高中低分类

这些文章的研究涉及方方面面，对一些现象得出了自己的一些观点，为此总结出53个方向，并想要获取这些文章对着53个方向的支持与否。

想要达成这个目的，可以采用prompt工程，将标题、摘要和关键词加提示词一并交给deepseekv3进行判断具体提示词如下

> 注意：prompt工程无需训练集和测试集，在确保prompt内容和大语言模型的能力值得信任的时候，就可以直接使用该模式得到结果。那该如何判断prompt内容和大语言模型的能力是否值得信任呢，prompt内容值得信任要看内容是否准确表达出自己的意思和目的，并且考虑到特殊情况和必要的强调信息；大语言模型的能力需要在自己的prompt跑出结果后人为的去判断，用于量化的指标可以是准确率和召回率，放到我们的例子里面准确率就是一个文章输出了43个方向，其中30个是文章应该有的，那么准确率就是30/43，召回率是一篇文章本来有33个方向，但是返回的43个方向里面只有30个是符合的，这时候召回率就是30/33，一般来说准召越高越好。

```txt
Please perform two tasks based on the following article content:
1. **Type determination** : Determine whether the article belongs to one or more of the following categories, based on whether my input mentions or is related to the following categories and descriptions.
  - cognitive_offload: Delegate memory or computational tasks to AI to reduce human burden
  - information_processing_aid: AI helps integrate and analyze a large amount of information
  - decision_heuristics: Rely on simplified rules provided by AI to make decisions
  - bias_mitigation_amplification: AI may alleviate or amplify human cognitive biases
  - ambiguity_reduction: AI provides clarification under uncertain information
  - situation_awareness: AI enhances overall understanding of the environment or task
  - cognitive_load_management: AI regulates cognitive load through prompting/support
  - bounded_rationality: The limitations of making decisions with limited information and abilities
  - anchoring_adjustment: Decision is influenced and adjusted by the initial anchor point
  - error_probability_estimation: Estimation of risk and error probability
  - preference_learning: AI learns and predicts user preferences
  - mind_perception: People endow AI with the perception of mind or intention
  - cognition_emotion_interaction: The interaction between emotions and cognition in decision-making
  - human_ai_interaction: The interaction process between humans and AI
  - human_machine_teaming: Human machine collaboration as a team to complete tasks
  - cooperation_competition_dynamics: Cooperation and competition between humans and machines
  - delegated_decision_making: Delegate some or all decisions to AI
  - responsibility_allocation: Responsibility attribution and accountability allocation between humans and machines
  - trust_calibration: Dynamically adjust trust in AI based on information or performance
  - trust_dynamics: The process of trust evolving over time and experience
  - trust_miscalibration: Excessive or insufficient trust in AI
  - trust_repair_strategies: Ways to repair trust after AI errors
  - algorithmic_attitudes: Appreciation, preference, or rejection attitude towards algorithms
  - automation_bias: Excessive reliance on automation recommendations
  - uncertainty_communication: Transmitting uncertainty information of AI
  - perceived_fairness: User perception of fairness in AI decision-making
  - value_alignment: Are AI goals aligned with human values
  - social_presence: The social presence demonstrated by AI in interaction
  - anthropomorphism_effects: The effect of viewing AI as a humanoid individual
  - levels_of_autonomy: Autonomous control allocation of human-machine at different levels
  - adaptive_automation: AI dynamically adjusts automation level based on context
  - automation_transparency: Is the AI decision-making process transparent and interpretable
  - error_recovery_support: AI provides error repair or remedial support
  - interaction_fluency: Is human-computer interaction natural and smooth
  - coordination_ability: The ability of AI to coordinate within a team
  - team_shared_awareness: Joint situational awareness between team members and AI
  - feedback_loops: A cyclic mechanism that influences trust or decision-making through feedback
  - task_crafting: Users redefine task methods based on AI
  - negative_work_rumination: Repeatedly reflecting on negative experiences in AI or tasks
  - approach_avoidance_dynamics: User's tendency to approach or avoid AI
  - ai_empathy_dynamics: Empathy demonstrated by AI and user perception of it
  - emotion_recognition: AI recognizes human emotions or imitates emotional expressions
  - affective_transition: The transfer effect of emotions between AI and humans
  - stress_anxiety_regulation: AI helps alleviate stress and anxiety
  - motivation_engagement: AI stimulates user motivation and engagement
  - psychological_safety: User's sense of security in interacting with AI
  - self_efficacy: Individual self-efficacy supported by AI
  - agency_restoration: User regain control and autonomy
  - moral_agency: The moral responsibility of AI or human-machine in decision-making
  - responsibility_gap: The gap of ambiguous responsibility in human-machine joint decision-making
  - ethical_dissonance: The conflict between AI decision-making and human ethics
  - ai_threat_perceptions: User perception of potential threats from AI
  - psychological_expansion: AI helps expand users' psychological or cognitive abilities


2. **Level determination**:
   Please determine the level of the category to which the current article belongs, and divide it into "high", "medium", and "low" from high to low. If there are multiple categories, provide the categories and corresponding levels separately.
  - Cognitive_offload: ['High: Almost completely dependent on AI processing tasks', 'Medium: Partially dependent (auxiliary+autonomous combination)', 'Low: Fully autonomous']
  - Information_decessing  - aid: ['High: AI significantly improves understanding/integration speed ',' Medium: AI assistance is limited ',' Low: No help or increases complexity ']
  - Decision_ceuristic: ['High: Highly dependent on AI to provide shortcuts', 'Medium: Combining heuristics and rationality', 'Low: Strict rationality']
  - Bias_mitigation_mamplification: ['High: AI significantly reduces/amplifies bias', 'Medium: limited impact', 'Low: no effect']
  - Ambiguity Reduction: ['High: AI explicitly reduces uncertainty ',' Medium: partially clarifies', 'Low: still ambiguous']
  - Situation  - awareness: ['High: Significant improvement in environmental understanding ',' Medium: Limited improvement ',' Low: No help or reduced perception ']
  - Cognitive  - load_management: ['High: significantly reduces cognitive burden ',' Medium: partially alleviates', 'Low: burden maintenance or increase']
  - Bounded  - rationality: ['High: severely constrained decision ',' Medium: partially constrained ',' Low: approaching rationality ']
  - AnchoringAdjustment: ['High: Clearly affected by initial anchor point ',' Medium: Partial adjustment ',' Low: Independent judgment ']
  - Error probability estimation: ['High: able to accurately estimate risk ',' Medium: rough estimate ',' Low: lack of risk awareness']
  - Preference_learning: ['High: AI/human preference for fast learning ',' Medium: limited learning ',' Low: no learning ']
  - Mind  - perception: ['High: strongly believes that AI has a mind ',' Medium: partially endows features', 'Low: completely as a tool']
  - Cognitive  - emotion  - interaction: ['High: Strong emotional influence on decision  - making ',' Medium: Partial influence ',' Low: Independent of emotions']
  - Human_ai_interaction: ['High: frequent and deep interaction ',' Medium: occasional interaction ',' Low: almost no interaction ']
  - Human_machine_ teaming: ['High: highly collaborative and complementary ',' Medium: limited collaboration ',' Low: no sense of teamwork ']
  - Cooperation_compatition  - d ynamics: ['High: Significant impact of cooperation/competition ',' Medium: Partial impact ',' Low: No effect ']
  - Delegated_decision_making: ['High: heavily delegated AI ',' Medium: partially delegated ',' Low: completely autonomous']
  - Responsibility_allocation: ['High: clear and reasonable responsibility ',' Medium: partially vague ',' Low: unclear or evasive ']
  - Trust_calibration: ['High: Precise dynamic adjustment of trust ',' Medium: Limited adjustment ',' Low: No adjustment ']
  - Trust_rynamics: ['High: trust fluctuates significantly over time ',' Medium: small changes', 'Low: stable and unchanged']
  - Trust_ciscalibration: ['High: Clearly excessive/insufficient trust ',' Medium: Mild mismatch ',' Low: Trust matching ']
  - Trust_depair_Strategies: ['High: Effectively fixes trust ',' Medium: Partially fixes', 'Low: Fix failed']
  - Algorithmic Attitudes: ['High: Strong Appreciation or Exclusion ',' Medium: Neutral Attitude ',' Low: Weak Attitude ']
  - Automation  - bias: ['High: blindly following AI suggestions', 'Medium: partial dependencies',' Low: independent judgment ']
  - Uncertainty_communication: ['High: Clearly Convey Uncertainty ',' Medium: Partially Convey ',' Low: Opaque ']
  - Perceived_fairness: ['High: strongly perceived fairness', 'Medium: generally fair', 'Low: unfair']
  - Value_alignment: ['High: highly consistent target ',' Medium: partially consistent ',' Low: severe conflict ']
  - Sociopresence: ['High: Strong Presence ',' Medium: Limited Presence ',' Low: Almost None ']
  - Anthropomorphism effects: ['High: Strongly endowing humanoid traits', 'Medium: Limited endowing', 'Low: Utilized']
  - Levels_of_ autonomy: ['High: AI fully controlled ',' Medium: shared control ',' Low: human fully controlled ']
  - Adaptive_automation: ['High: Highly Adaptive ',' Medium: Limited Adaptive ',' Low: Fixed ']
  - Automation  - transparency: ['High: Transparent Decision Process', 'Medium: Partially Transparent', 'Low: Black Box']
  - Erre_recovery_Support: ['High: Effectively fixes errors', 'Medium: Limited support', 'Low: No support']
  - Interaction_fluency: ['High: Smooth and Natural ',' Medium: Occasionally Smooth ',' Low: Stuck ']
  - Coordination Ability: ['High: Highly Coordinated ',' Medium: Partially Coordinated ',' Low: Lack of Coordinated ']
  - Team_Shared_awareness: ['High: highly consistent understanding ',' Medium: partially consistent ',' Low: serious disagreement ']
  - Feedback_rops: ['High: Strong feedback drives change ',' Medium: Limited feedback ',' Low: No effect ']
  - Task_craft: ['High: Actively redefine tasks', 'Medium: Partial adjustments',' Low: No adjustments']
  - Negative work_rumination: ['High: Strong Reflection on Failure ',' Medium: Limited Reflection ',' Low: No Reflection ']
  - Approach'avoidanced_dynamics: ['High: obvious approach/avoidance ',' Medium: partial performance ',' Low: no inclination ']
  - Ai_ empathy_dynamics: ['High: AI strongly displays empathy&user perception ',' Medium: partial display/perception ',' Low: no empathy ']
  - Emotion  - recognition: ['High: Accurate emotion recognition ',' Medium: Limited recognition ',' Low: Recognition failure ']
  - Affective_transition: ['High: Strong emotional transfer ',' Medium: Limited transfer ',' Low: No transfer ']
  - Stress_anxiety_regulation: ['High: significantly reduces stress anxiety ',' Medium: partially relieved ',' Low: no relief or exacerbation ']
  - Motivation_degagment: ['High: Highly engaged and proactive ',' Medium: Moderately engaged ',' Low: Indifferent and unmotivated ']
  - Psychologic_Safety: ['High: completely safe ',' Medium: partially safe ',' Low: no sense of security ']
  - Self efficacy: 'High: Strong Self Efficacy', 'Medium: Partial Confidence', 'Low: No Confidence'
  - Agey_restoration: ['High: Strong sense of control restored ',' Medium: Partial recovery ',' Low: No sense of control ']
  - Mortal  - agency: ['High: Actively taking on moral decisions', 'Medium: Limited taking on responsibilities',' Low: No taking on responsibilities']
  - Responsibility_cap: ['High: Serious ambiguity of responsibility ',' Medium: Partial ambiguity ',' Low: Clear ']
  - Ethical  - dissonance: ['High: Strong Ethical Conflict ',' Medium: Limited Conflict ',' Low: No Conflict ']
  - Ai_threat_perceptions: ['High: Strong threat ',' Medium: Partial threat ',' Low: No threat ']
  - Psychologis_expansion: ['High: Significant Scalability ',' Medium: Limited Scalability ',' Low: No Scalability ']

3. **Remarks**:
   - It is normal for a piece of data to belong to multiple categories, but one category can only belong to one of high, medium, or low.
   - The content in the output str should be enclosed in single quotes to prevent parsing errors in the json format.

**Output requirements**:
  - Output in JSON format, within one line
  - Contains four fields:
  "Classification": A list that stores the results of type judgments,
  "Levels": A dictionary that stores the corresponding levels of judgment results,
  "Reason": A dictionary that stores the reasons for the corresponding judgment results, requiring input original sentences that support classification and classification levels, The length of the return is unlimited,
  "Brief": A brief introduction to the article.

**Processing examples**:
Input Title: "Title of the article"
Input Summary:"Abstract of the article"
Output:
```json
{
  "classification": ["class1", "class2"],
  "levels": {"class1":"low","class2":"high"}
  "reason": {"class1":"reason", "class2":"reason"},
  "brief": xxx
}```

**Input**:
Title: <TITLE>
Abstract: <ABSTRACT>
Kerwords: <KEYWORDS>
```

prompt可以定向激活大语言模型的能力，使其按照prompt的指令形式，完成定向的任务，也给出规定格式的答案，调用模型的代码如下

```python
import json
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import threading
from queue import Queue

# 读取数据
df = pd.read_csv("../initialize/filtered_prediction(1096).csv")

# 创建新列来存储结果
df["classification"] = ""
df["levels"] = ""
df["reason"] = ""
df["brief"] = ""

client = OpenAI(
    api_key="sk-73b1e9510c504a54a5eeabeadde2d51e",  # ⚠️ 替换为你自己的
    base_url="https://api.deepseek.com"
)


def req_v3(promptxxxxx):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": promptxxxxx}]
        )
        reply = response.choices[0].message.content.strip()
        return reply
    except Exception as e:
        print("⚠️ DeepSeek API 调用失败:", e)
        return "Unknown"


def get_prompt(title, abstract, keywords, prompt_info):
    """ get_prompt """
    if not isinstance(keywords, str):
        keywords = ""
    prompt_tmp = prompt_info
    prompt_tmp = prompt_tmp.replace("<TITLE>", title)
    prompt_tmp = prompt_tmp.replace("<ABSTRACT>", abstract)
    prompt_tmp = prompt_tmp.replace("<KEYWORDS>", keywords)
    return prompt_tmp


def make_offline_input(title, abstract, keywords, promptxxx):
    prompt_infos = get_prompt(title, abstract, keywords, promptxxx)
    req_dict = {
        "request_body": {
            "system": "",
            "messages": [
                {
                    "role": "user",
                    "content": prompt_infos
                }
            ],
            "top_p": 0.8,
            "temperature": 1,
            "penalty_score": 1
        }
    }

    out_list = json.dumps(req_dict, ensure_ascii=False)
    return out_list


# 处理单个数据项的函数
def process_item(args):
    i, row, prompts = args
    title_info = row["Title"]
    abstract_info = row["Abstract"]
    keywords_info = row["Keywords"]

    # 生成提示
    prompt = get_prompt(title_info, abstract_info, keywords_info, prompts)

    # 调用API
    result = req_v3(prompt)

    # 尝试解析JSON结果
    try:
        result_data = json.loads(result.replace("json", "").replace("`", '').strip())
        classification = result_data.get("classification", "Unknown")
        levels = result_data.get("levels", "Unknown")
        reason = result_data.get("reason", "Unknown")
        brief = result_data.get("brief", "Unknown")
        result = result
    except json.JSONDecodeError:
        # 如果结果不是JSON格式，直接存储原始结果
        classification = "Parse Error"
        levels = "Parse Error"
        reason = result
        brief = "Parse Error"
        result = result

    return i, classification, levels, reason, brief, result


# 读取提示模板
with open("v4_full.txt", 'r', encoding='utf-8') as f:
    prompts = f.read()

# 准备任务参数
tasks = [(i, row, prompts) for i, row in df.iterrows()]

# 使用线程池处理数据，但保持顺序
max_workers = 10  # 根据API限制调整线程数

# 方法1: 使用map保持顺序
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(tqdm(executor.map(process_item, tasks), total=len(tasks)))

# 按原始顺序更新DataFrame
for i, classification, levels, reason, brief, result in results:
    df.at[i, "classification"] = classification
    df.at[i, "levels"] = levels
    df.at[i, "reason"] = reason
    df.at[i, "brief"] = brief
    df.at[i, "result"] = result  # 标记为已处理

    # 每处理100条记录保存一次
    if (i + 1) % 100 == 0:
        df.to_csv("53_100_classification.csv", index=False)
        print(f"已保存前 {i + 1} 条结果")

# 最终保存
df.to_csv("53_full_classification.csv", index=False)
print("处理完成，结果已保存到 53_full_classification.csv")
```



下面是53个方向的频次统计图

![Statistical_Chart_of_Frequency_of_Research_Direction_and_Viewpoints](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/Statistical_Chart_of_Frequency_of_Research_Direction_and_Viewpoints.png)

**共现热力图** → 谁跟谁“经常一起出现”。

![heatmap_cooccurrence](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/heatmap_cooccurrence.png)

**相似度热力图** → 谁跟谁“整体特征相似”。

![heatmap_similarity](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/heatmap_similarity.png)

**聚类热力图** → 在相似度的基础上，把节点分群，更清晰地看到潜在的结构模块。

![clustermap_similarity_left_cbar](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/clustermap_similarity_left_cbar.png)

链式关系图

![chain](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/屏幕截图 2025-09-06 190524.png)



九宫格图

![nine](https://raw.githubusercontent.com/huangshuai-ustc/images/main/trust/nine.png)