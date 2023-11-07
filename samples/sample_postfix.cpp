#include <iostream>
#include <string>
#include <sstream>
#include "postfix.h"

using namespace std;

string get_postfix_as_string(const TArithmeticExpression& expr)
{
    const auto& postfix = expr.get_postfix();
    const size_t size = postfix.size();

    stringstream ss;
    for (size_t i = 0; i < size - 1; i++)
    {
        ss << postfix[i] << ' ';
    }
    ss << postfix[size - 1];

    return ss.str();
}

int main()
{
    setlocale(LC_ALL, "Russian");
    setlocale(LC_NUMERIC, "en_US.UTF-8");
    /////

    string infix;
    cout << "Введите арифметическое выражение: ";
    cin >> infix;
    cout << endl;

    TArithmeticExpression expr(infix);
    cout << "Арифметическое выражение: " << expr.get_infix() << endl;
    cout << "Постфиксная форма: '" << get_postfix_as_string(expr) << "'" << endl;
    cout << endl;

    auto variables = expr.get_variables();
    std::map<std::string, double> values;
    for (const auto& var : variables)
    {
        cout << " Введите значение для '" << var << "': ";
        cin >> values[var];
    }
    //
    auto func_names = expr.get_functions();
    std::map<std::string, std::shared_ptr<TArithmeticExpressionFunction>> functions;
    for (const auto& fun : func_names)
    {
        cout << " Введите формулу для '" << fun << "'(x): ";
        cin >> infix;
        functions[fun] = std::make_shared<TExplicitArithmeticExpressionFunction>(TArithmeticExpression(infix));
    }
    cout << endl;

    double res = expr.calculate(values, functions);
    cout << "Результат: " << res << endl;

    return 0;
}
