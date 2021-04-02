# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from durable.lang import *
def AnalysisRules():
    with ruleset('Analysis'):
        @when_all(m.purpose == '실험분석')
        def test_anal(c):
            c.assert_fact({'purpose': c.m.purpose, 'predicate': '중', 'category': '통계분석',
                           'data_kind' : c.m.data_kind,'var_group': c.m.var_group, 'etc':'',
                           'relation':c.m.relation, 'SampleSize':c.m.SampleSize})

        @when_all(m.purpose == '품질관리')
        def factory_anal(c):
            c.assert_fact({'purpose': c.m.purpose, 'predicate': '중', 'category': '관리도',
                           'data_kind': c.m.data_kind, 'var_group': c.m.var_group,'etc':'',
                           'relation':c.m.relation, 'SampleSize':c.m.SampleSize})

        @when_all((m.category =='통계분석') & (m.data_kind == 'Number') & (m.var_group == 1) & (m.relation == 'N'))
        def frequency_analysis(c): # 빈도 분석
            c.assert_fact({'purpose': c.m.object, 'predicate': '은', 'object': '빈도분석',
                           'category':c.m.category,'etc':'데이터 종류는 빈도이고 변수는 '+str(c.m.var_group)+'개이다'})

        @when_all((m.category == '통계분석') & (m.data_kind == 'Number') & (m.var_group > 1) & (m.relation == 'N'))
        def cross_analysis(c):  # 교차 분석
            c.assert_fact({'purpose': c.m.object, 'predicate': '은', 'object': '교차분석','category':c.m.category,
                           'etc': '데이터 종류는 빈도이고 변수는 ' + str(c.m.var_group) + '개이다'})

        @when_all((m.category == '통계분석') & (m.data_kind == 'Data') & (m.var_group == 2) & (m.relation == 'N'))
        def average_analysis(c):  # 평균 분석
            c.assert_fact({'purpose': c.m.object, 'predicate': '은', 'object': '평균분석','category':c.m.category,
                           'etc': '데이터 종류는 점수이고 변수는 ' + str(c.m.var_group) + '개이다'})

        @when_all((m.category == '통계분석') & (m.data_kind == 'Data') & (m.var_group >= 3) & (m.relation == 'N'))
        def anova_analysis(c):  # 변량 분석(ANOVA)
            c.assert_fact({'purpose': c.m.object, 'predicate': '은', 'object': '변량분석','category':c.m.category,
                           'etc': '데이터 종류는 점수이고 변수는 ' + str(c.m.var_group) + '개이다'})

        @when_all((m.category == '통계분석') & (m.data_kind == 'Data') & (m.var_group == 2) & (m.relation == 'Y'))
        def correlation_analysis(c):  # 상관 분석
            c.assert_fact({'purpose': c.m.object, 'predicate': '은', 'object': '상관분석','category':c.m.category,
                           'etc': '데이터 종류는 점수이고 변수는 ' + str(c.m.var_group) + '개이다. 변수간의 관계성을 분석한다'})

        @when_all((m.category == '통계분석') & (m.data_kind == 'Data') & (m.var_group > 1) & (m.relation == 'dependent'))
        def regression_analysis(c):  # 회귀 분석
            c.assert_fact({'purpose': c.m.object, 'predicate': '은', 'object': '회귀분석','category':c.m.category,
                           'etc': '데이터 종류는 점수이고 변수는 ' + str(c.m.var_group) + '개이다. 변수간의 관계성을 분석한다'})

        @when_all((m.category == '관리도') & (m.data_kind == 'Data') & (m.var_group >= 5))
        def XBar_R_Control(c):  # XBar R 관리도
            c.assert_fact({'purpose': c.m.object, 'predicate': '은', 'object': 'XBar-R','category':c.m.category,
                           'etc': '데이터 종류는 점수이고 군의 크기는 ' + str(c.m.var_group) + '개이다'})

        @when_all((m.category == '관리도') & (m.data_kind == 'Data') & (m.var_group < 5) & (m.var_group > 1) )
        def XBar_S_Control(c):  # XBar S 관리도
            c.assert_fact({'purpose': c.m.object, 'predicate': '은', 'object': 'XBar-S','category':c.m.category,
                           'etc': '데이터 종류는 점수이고 군의 크기는 ' + str(c.m.var_group) + '개이다'})

        @when_all((m.category == '관리도') & (m.data_kind == 'Data') & (m.var_group == 1))
        def IMR_Control(c):  # IMR 관리도
            c.assert_fact({'purpose': c.m.object, 'predicate': '은', 'object': 'IMR','category':c.m.category,
                           'etc': '데이터 종류는 점수이고 군의 크기는 ' + str(c.m.var_group) + '개이다'})

        @when_all((m.category == '관리도') & (m.data_kind == 'Bad_Rate') & (m.var_group == 1) & (m.SampleSize == '일정'))
        def nP_Control(c): # np 관리도
            c.assert_fact({'purpose': c.m.object, 'predicate': '은', 'object': 'nP','category':c.m.category,
                           'etc': '데이터 종류는 불량률이고 표본 크기는 ' + c.m.SampleSize + '하다'})

        @when_all((m.category == '관리도') & (m.data_kind == 'Bad_Rate') & (m.var_group == 1) & (m.SampleSize == '비일정'))
        def p_Control(c):  # p 관리도
            c.assert_fact({'purpose': c.m.object, 'predicate': '은', 'object': 'p','category':c.m.category,
                           'etc': '데이터 종류는 불량률이고 표본 크기는 ' + c.m.SampleSize + '하다'})

        @when_all((m.category == '관리도') & (m.data_kind == 'Bad_Per_Unit') & (m.var_group == 1) & (m.SampleSize == '일정'))
        def c_Control(c):  # c관리도
            c.assert_fact({'purpose': c.m.object, 'predicate': '은', 'object': 'c','category':c.m.category,
                           'etc': '데이터 종류는 결점 수이고 표본 크기는 ' + c.m.SampleSize + '하다'})

        @when_all((m.category == '관리도') & (m.data_kind == 'Bad_Per_Unit') & (m.var_group == 1) & (m.SampleSize == '비일정'))
        def u_Control(c):  # u 관리도
            c.assert_fact({'purpose': c.m.object, 'predicate': '은', 'object': 'u', 'category':c.m.category,
                           'etc': '데이터 종류는 결점 수이고 표본 크기는 ' + c.m.SampleSize + '하다'})

        @when_any((m.object == 'u')|(m.object == 'c'))
        def Poisson (c):  # 포아송 분포
            c.assert_fact({'purpose': "", 'predicate': '는', 'object': c.m.object, 'category': "포아송 분포",
                       'etc': ''})

        @when_any((m.object == 'nP') | (m.object == 'p'))
        def Binomial_distribution(c):  # 이항분포
            c.assert_fact({'purpose': "", 'predicate': '는', 'object': c.m.object, 'category': "이항 분포",
                           'etc': ''})

        @when_any( (m.object == 'IMR') | (m.object =='XBar-R')| (m.object =='XBar-S'))
        def Normal_distribution(c):  # 정규분포
            c.assert_fact({'purpose': "", 'predicate': '는', 'object': c.m.object, 'category': "정규 분포",
                           'etc': ''})
        @when_all(+m.object)
        def output(c):
            print(' {0} {1} {2}이다.{3}'.format(c.m.object, c.m.predicate, c.m.category,c.m.etc))

    assert_fact('Analysis', {'purpose': '실험분석', 'data_kind': 'Number', 'var_group': 1, 'relation' : "N",'SampleSize':'None'})# 빈도 분석
    assert_fact('Analysis', {'purpose': '실험분석', 'data_kind': 'Number', 'var_group': 2, 'relation' : "N",'SampleSize':'None'})# 교차 분석
    assert_fact('Analysis', {'purpose': '실험분석', 'data_kind': 'Data', 'var_group': 2, 'relation': "N",'SampleSize': 'None'})  # 평균분석
    assert_fact('Analysis', {'purpose': '실험분석', 'data_kind': 'Data', 'var_group': 3, 'relation' : "N",'SampleSize':'None'})# 변량 분석
    assert_fact('Analysis', {'purpose': '실험분석', 'data_kind': 'Data', 'var_group': 2, 'relation': "Y",'SampleSize':'None'}) # 상관 분석
    assert_fact('Analysis', {'purpose': '실험분석', 'data_kind': 'Data', 'var_group': 2, 'relation': "dependent",'SampleSize':'None'}) # 회귀 분석
    assert_fact('Analysis', {'purpose': '품질관리', 'data_kind': 'Data', 'var_group': 5, 'relation': "N",'SampleSize':'None'}) # XBar-R
    assert_fact('Analysis', {'purpose': '품질관리', 'data_kind': 'Data', 'var_group': 4, 'relation': "N",'SampleSize':'None'}) # XBar -S

    assert_fact('Analysis', {'purpose': '품질관리', 'data_kind': 'Data', 'var_group': 1, 'relation': "N",'SampleSize':'None'}) #I-MR
    assert_fact('Analysis', {'purpose': '품질관리', 'data_kind': 'Bad_Rate', 'var_group': 1, 'relation': "N", 'SampleSize': '일정'})#np
    assert_fact('Analysis', {'purpose': '품질관리', 'data_kind': 'Bad_Rate', 'var_group': 1, 'relation': "N", 'SampleSize': '비일정'})#p

    assert_fact('Analysis',{'purpose': '품질관리', 'data_kind': 'Bad_Per_Unit', 'var_group': 1, 'relation': "N", 'SampleSize': '일정'}) # c
    assert_fact('Analysis',{'purpose': '품질관리', 'data_kind': 'Bad_Per_Unit', 'var_group': 1, 'relation': "N", 'SampleSize': '비일정'}) #u

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    AnalysisRules()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
