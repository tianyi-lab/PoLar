"""Modified from https://github.com/OpenBMB/OlympiadBench"""

import math
import re
import sys

import sympy as sp
from sympy import Eq, Pow, simplify, sympify
from sympy.parsing.latex import parse_latex

OB_ANS_TYPE_ID2ZH = {
    "Numerical": "数值",
    "Expression": "表达式",
    "Equation": "方程",
    "Interval": "区间",
}
OB_ANS_TYPE_ID2EN = {
    "Numerical": "a numerical value",
    "Expression": "an expression",
    "Equation": "an equation",
    "Interval": "an interval",
}


def get_single_answer_type_text(answer_type, is_chinese):
    if "-" in answer_type:  # No need now
        answer_type = answer_type[: answer_type.find("-")]
    for t in ["Numerical", "Expression", "Equation", "Interval"]:
        if t in answer_type:
            if is_chinese:
                return OB_ANS_TYPE_ID2ZH[t]
            else:
                return OB_ANS_TYPE_ID2EN[t]
    exit(f"Error parsing answer type {answer_type}!")


def get_answer_type_text(answer_type, is_chinese, multiple_answer):
    if ("Need_human_evaluate" in answer_type) or (
        "Tuple" in answer_type
    ):  # 'Tuple' has various meanings in different context, such as position or values of a series of variable, so it may lead to confusion to directly use 'tuple' in the prompt.
        full_answer_text = ""
    else:
        if not multiple_answer:
            answer_text = get_single_answer_type_text(answer_type, is_chinese)
            if is_chinese:
                full_answer_text = f"，答案类型为{answer_text}"
            else:
                full_answer_text = (
                    f"The answer of The problem should be {answer_text}. "
                )
        else:
            if "," not in answer_type:  # Same answer type for all answers
                answer_text = get_single_answer_type_text(answer_type, is_chinese)
                if is_chinese:
                    full_answer_text = f"，题目有多个答案，答案类型均为{answer_text}"
                else:
                    full_answer_text = f"The problem has multiple answers, each of them should be {answer_text}. "
            else:
                answer_types = answer_type.split(",")
                answer_types = [
                    get_single_answer_type_text(t, is_chinese) for t in answer_types
                ]
                if len(set(answer_types)) == 1:
                    answer_text = answer_types[0]
                    if is_chinese:
                        full_answer_text = (
                            f"，题目有多个答案，答案类型均为{answer_text}"
                        )
                    else:
                        full_answer_text = f"The problem has multiple answers, each of them should be {answer_text}. "
                else:
                    if is_chinese:
                        answer_text = "、".join(answer_types)
                        full_answer_text = (
                            f"，题目有多个答案，答案类型分别为{answer_text}"
                        )
                    else:
                        answer_text = ", ".join(answer_types)
                        full_answer_text = f"The problem has multiple answers, with the answers in order being {answer_text}. "
    return full_answer_text


def make_prompt(
    question: str, is_chinese: bool, is_math: bool, is_theorem_proving: bool
) -> str:
    if is_chinese:
        subject_content = "数学" if is_math else "物理"
        if is_theorem_proving:
            prompt = f"以下是中国{subject_content}竞赛中的证明题。请根据题目的要求，运用逻辑推理及常用定理证明题目中的命题。证明过程中使用的变量和公式请使用LaTeX格式表示。"
        else:
            answer_type_text = get_answer_type_text(
                question["answer_type"],
                is_chinese=True,
                multiple_answer=question["is_multiple_answer"],
            )
            if question["is_multiple_answer"]:
                multiple_answer_text = "\\boxed{用英文逗号连接的多个答案}"
            else:
                multiple_answer_text = "\\boxed{答案}"
            unit_text = ""
            if question["unit"]:
                multiple_answer_text += "(单位)"
                unit_text = "，注意答案的单位不要放在\\boxed{}中"
            prompt = f"以下是中国{subject_content}竞赛中的解答题{answer_type_text}。请根据题目的要求和所提供的信息计算得出答案。解答过程和结果中使用的变量和公式请使用LaTeX格式表示。请在最后以“所以最终答案是{multiple_answer_text}。”显式给出结果{unit_text}。"
    else:
        subject_content = "Math" if is_math else "Physics"
        if is_theorem_proving:
            prompt = f"The following is a theorem proving problem from an International {subject_content} competition. Please use logical reasoning and common theorems to prove the proposition in the problem according to the given requirements. Please use LaTeX format to represent the variables and formulas used in the proof."
        else:
            if question["is_multiple_answer"]:
                multiple_answer_text = "\\boxed{multiple answers connected with commas}"
            else:
                multiple_answer_text = "\\boxed{answer}"
            unit_text = ""
            if question["unit"]:
                multiple_answer_text += "(unit)"
                unit_text = ", note that the unit of the answer should not be included in \\boxed{}"
            answer_type_text = get_answer_type_text(
                question["answer_type"],
                is_chinese=False,
                multiple_answer=question["is_multiple_answer"],
            )
            prompt = f'The following is an open-ended problem from an International {subject_content} competition. {answer_type_text}Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results. Please end your solution with "So the final answer is {multiple_answer_text}." and give the result explicitly{unit_text}.'
    return prompt


def make_query(sample: dict[str, dict[str, str]], ds_filename: str) -> str:
    is_theorem_proving: bool = "TP" in ds_filename
    is_math: bool = "maths" in ds_filename
    is_chinese: bool = "zh" in ds_filename

    prompt = make_prompt(sample, is_chinese, is_math, is_theorem_proving)
    query = prompt
    if not is_math and "context" in sample.keys():
        query += "\n" + sample["context"]
    query += "\n" + sample["question"]

    return query


# sys.set_int_max_str_digits(1000000)


class OlympiadMathJudger:
    def __init__(self):
        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.pi = parse_latex("\\pi")
        self.precision = 1e-8

    def split_by_comma(self, expr: str):
        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char == "(" or char == "[":
                in_bracket_num += 1
            elif char == ")" or char == "]":
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())

        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list):
        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)

        return new_expr_list

    def judge(self, expression1, expression2, precision=1e-8):
        # (默认 expression1 为 Ground_Truth)
        precision = precision if type(precision) == list else [precision]

        try:
            expression1, expression2 = self.preprocess(expression1, expression2)
        except:
            return False
        if expression1 == expression2:
            # print("原生相等")
            return True

        # 去除字符串中的中文字符，因为上面已经判断过了类似回答为"能"或"不能"的含有中文字符的回答情况
        expression1 = re.sub(r"[\u4e00-\u9fff]+", "", expression1)
        expression2 = re.sub(r"[\u4e00-\u9fff]+", "", expression2)

        expression1 = self.split_by_comma(expression1)
        expression2 = self.split_by_comma(expression2)

        temp_list1 = self.trans_plus_minus_sign(expression1)
        temp_list2 = self.trans_plus_minus_sign(expression2)

        # 设计误差值列表
        if len(precision) <= 1:
            precision = precision * len(temp_list1)

        if len(temp_list1) != len(temp_list2):
            return False

        # 判断两个列表中的元素是否可以两两配对，并且两两相等，由此支持多个回答的比较
        idx = -1
        while len(temp_list1) != 0:
            idx = (idx + 1) % len(temp_list1)

            item1 = temp_list1[idx]
            self.precision = precision[idx]
            # print(self.precision)

            for item2 in temp_list2:
                if self.is_equal(item1, item2):
                    temp_list1.remove(item1)
                    temp_list2.remove(item2)
                    precision.remove(self.precision)
                    break
            else:
                # If we didn't break from the inner loop, it means no match was found
                return False

        # If all elements are matched and removed, the lists can be paired
        return True

    def is_interval(self, epr):
        return epr.startswith(("(", "[")) and epr.endswith((")", "]"))

    # 在进行数值计算前，需要将sympy中的pi符号替换为pi的近似数值
    def sympy_sub_pi(self, expression_sympy):
        return expression_sympy.subs(self.pi, math.pi)

    # 默认第一个表达式是 ground_truth
    def is_equal(self, expression1, expression2):
        if expression1 == expression2 and expression1 != "" and expression2 != "":
            # print("原生等价")
            return True

        # 先判断是否是两个区间，是的话进行判断相等，不相等则返回 False
        if self.is_interval(expression1) and self.is_interval(expression2):
            try:
                if self.interval_equal(expression1, expression2):
                    # print("区间等价")
                    return True
            except:
                return False

        # 再判断是否在数值上相等
        try:
            if self.numerical_equal(expression1, expression2):
                # print("数值等价")
                return True
        except:
            pass

        # 再判断是否是表达式相等
        try:
            if self.expression_equal(expression1, expression2) and not (
                "=" in expression1 and "=" in expression2
            ):
                # print("表达式等价")
                return True
        except:
            pass

        # 再判断是否是等式相等
        try:
            if self.equation_equal(expression1, expression2):
                # print("等式等价")
                return True
        except:
            pass

        return False

    # 判断两个数值在误差允许范围内是否相等
    def numerical_equal(
        self, expression1: str, expression2: str, include_percentage: bool = True
    ):
        """
        (默认 expression1 为 Ground_Truth)
        函数: 判读两个数值是否在误差允许范围内相等
        步骤1: 将可能出现的百分号的情况包含进来
        步骤2: 使用 math.isclose 函数判断是否相等
        """
        reference = float(expression1)
        prediction = float(expression2)

        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]

        for item in gt_result:
            # if isclose(item, prediction, abs_tol=self.precision, rel_tol=0):
            if abs(item - prediction) <= self.precision * 1.01:
                return True
        return False

    def expression_equal(self, exp1, exp2):
        """
        (默认 expression1 为 Ground_Truth)
        函数: 判断两个表达式是否在数学意义上等价
        步骤1: 提取表达式, 防止有的模型会给出"x=1"而不是"1"
        步骤2: 使用 sympy 库进行等价判断
        """

        # 只提取等号右边的表达式，一般左边是所求的量
        def extract_expression(expression):
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()

        exp1 = extract_expression(exp1)
        exp2 = extract_expression(exp2)

        # 将表达式转换为 sympy 中能够进行处理的格式
        expr1_sym = sympify(parse_latex(exp1))
        expr2_sym = sympify(parse_latex(exp2))

        if expr1_sym == expr2_sym:
            return True
        else:
            expr1_sym = self.sympy_sub_pi(expr1_sym)
            expr2_sym = self.sympy_sub_pi(expr2_sym)
            # 如果输入的表达式可以计算出具体数值的话，则将其进行数值计算的比较

            if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (
                not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)
            ):
                return False
            elif not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
                try:
                    if not (
                        self.can_compute_power(expr1_sym)
                        and self.can_compute_power(expr2_sym)
                    ):
                        # print(
                        #     f'These two number can not be calculated by current computer for: "{str(expr1_sym)}" and "{str(expr2_sym)}"'
                        # )
                        # slow
                        return False

                    if (
                        abs(expr1_sym.evalf() - expr2_sym.evalf())
                        <= self.precision * 1.01
                    ):
                        return True
                    else:
                        return False
                except:
                    return False
            else:
                try:
                    simplified_expr = simplify(expr1_sym - expr2_sym)

                    num_value = simplified_expr.evalf()

                    return abs(num_value) < 1e-3
                except:
                    return False

    def equation_equal(self, expression1, expression2):
        """
        (默认 expression1 为 Ground_Truth)
        函数: 判断两个方程是否在数学意义上等价
        步骤1: 将一个方程/等式化简为标准方程, 即等式的右边严格等于0, 接下来只需要判断两个等式的左边是否"等价"
        步骤2: 使用 sympy 库计算两个等式左边的商, 如果这个商或者这个商的倒数为整数, 那么数学意义上我们可以推导出这两个方程等价👌
        """

        # 将等式的右边都移到左边，并返回一个 sympy 格式的表达式
        def simplify_equation(latex_eq):
            # 分割等式的左边和右边
            lhs, rhs = latex_eq.split("=")

            # 使用 latex2sympy 解析 LaTeX 表达式
            lhs_expr = parse_latex(lhs)
            rhs_expr = parse_latex(rhs)

            # 创建等式对象
            equation = Eq(lhs_expr, rhs_expr)

            # 化简等式：将等式右边移到左边
            simplified_eq = simplify(equation.lhs - equation.rhs)

            return simplified_eq

        expr1_sym = simplify_equation(expression1)
        expr2_sym = simplify_equation(expression2)

        division_result_1 = simplify(expr1_sym / expr2_sym)
        division_result_2 = simplify(expr2_sym / expr1_sym)

        # 如果两个方程转换后的式子相除为整数 且非零，则根据推导可知这两个方程等价
        if (division_result_1.is_Integer and division_result_1 != 0) or (
            division_result_2.is_Integer and division_result_2 != 0
        ):
            return True
        else:
            return False

    def interval_equal(self, expression1, expression2):
        """
        函数: 判断两个区间是否在数学意义上等价
        步骤1: 简化区间的表达式, 去除无关的符号比如"\left", "\right", 同时将可能出现的"x \in"删去
        步骤2: 对比两个区间的左右符号、中间出现的数学表达式等是否一致
        """

        def compare_two_interval(inter1, inter2):

            # 首先比较两边的括号是否一致，一致的话再进行下一步比较
            if inter1[0] != inter2[0] or inter1[-1] != inter2[-1]:
                return False

            inter1 = inter1.strip("[]()")
            inter2 = inter2.strip("[]()")

            # 分割区间的左右部分
            items_1 = inter1.split(",")
            items_2 = inter2.split(",")

            for item_1, item_2 in zip(items_1, items_2):
                if not self.expression_equal(item_1, item_2):
                    return False
            return True

        interval1 = expression1
        interval2 = expression2

        if interval1 == interval2:
            return True
        else:
            inter_list1 = interval1.split("\\cup")
            inter_list2 = interval2.split("\\cup")

            if len(inter_list1) != len(inter_list2):
                return False
            else:
                for inter1, inter2 in zip(inter_list1, inter_list2):
                    if not compare_two_interval(inter1, inter2):
                        return False
                return True

    def preprocess(self, expression1, expression2):

        # 尝试捕获box中的内容，如果有多个则以逗号相连返回，如果一个都没有，则报错
        def extract_boxed_content(latex_str):
            # 查找所有的 \boxed{...} 结构
            boxed_matches = re.finditer(r"\\boxed{", latex_str)
            results = ""

            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1

                # 从 \boxed{ 之后开始搜索，直到找到对应的闭合括号
                while stack > 0 and end_index < len(latex_str):
                    if latex_str[end_index] == "{":
                        stack += 1
                    elif latex_str[end_index] == "}":
                        stack -= 1
                    end_index += 1

                if stack == 0:
                    # 提取 \boxed{} 内部的内容
                    content = latex_str[start_index : end_index - 1]
                    results += content + ","
                else:
                    # 如果括号没有正确闭合，则返回错误信息
                    raise ValueError("Mismatched braces in LaTeX string.")

            # 如果没有匹配到'\boxed{}'字符，则默认提取有内容的文字最后一行中的所有公式部分
            if results == "":
                last_line_ans = latex_str.strip().split("\n")[-1]
                dollar_pattern = r"\$(.*?)\$"
                answers = re.findall(dollar_pattern, last_line_ans)

                if answers:
                    for ans in answers:
                        results += ans + ","
                else:
                    results = latex_str

            return results

        def sepcial_symbol_replace(expression):
            if "\\in " in expression:
                expression = expression.split("\\in ")[1]

            # 进行特殊字符的替换，这些字符都不影响latex的解析，属于美观/修饰性字符
            for signal in self.special_signal_map:
                expression = expression.replace(signal, self.special_signal_map[signal])

            expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")

            pattern = r"\\(?:mathrm|mathbf)\{~?([^}]*)\}"
            expression = re.sub(pattern, r"\1", expression)

            return expression

        exp1, exp2 = extract_boxed_content(expression1), extract_boxed_content(
            expression2
        )
        exp1, exp2 = sepcial_symbol_replace(exp1), sepcial_symbol_replace(exp2)

        return exp1, exp2

    def can_compute_power(self, expr):
        """
        Check if the power expression can be computed.

        Parameters:
        expr (sympy expression): The expression to check.

        Returns:
        bool: True if the expression can be computed, False otherwise.
        """
        # Check if the expression is a power expression
        if isinstance(expr, Pow):
            # Extract the base and the exponent
            base, exp = expr.as_base_exp()

            # Check if the base and the exponent are numbers
            if base.is_number and exp.is_number:
                # Set a threshold for the maximum size of the exponent
                MAX_EXP = 1000  # This threshold can be adjusted based on the computing environment

                # Check if the exponent is greater than the threshold
                if abs(exp.evalf()) > MAX_EXP:
                    return False
                else:
                    return True
            else:
                # If the base or the exponent is not a number, we cannot compute the power
                return False
        else:
            # If the expression is not a power expression, return True as it is not the case we are checking for
            return True
