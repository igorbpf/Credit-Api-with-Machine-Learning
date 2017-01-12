from flask import Flask
from flask_restplus import Api, fields, Resource
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

from werkzeug.contrib.fixers import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)

api = Api(app,
        version='1.0',
        title='Credit Prediction API',
        description='Machine Learning for Credit Prediction')

mt = api.namespace('approve_credit',
    description='Credit operations')

parser = api.parser()
parser.add_argument('RevolvingUtilizationOfUnsecuredLines',
                    type=float,
                    required=True,
                    help="Total balance on credit cards and pesonal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits",
                    location='form')

parser.add_argument('Age',
                    type=int,
                    required=True,
                    help='Age of borrower in years',
                    location='form')

parser.add_argument('NumberOfTime30-59DaysPastDueNotWorse',
                    type=int,
                    required=True,
                    help='Number of times borrower has been 30-59 days past due but no worse in the last 2 years.',
                    location='form')

parser.add_argument('DebtRatio',
                    type=float,
                    required=True,
                    help='Monthly debt payments, alimony,living costs divided by monthy gross income',
                    location='form')

parser.add_argument('MonthlyIncome',
                    type=float,
                    required=True,
                    help='Monthly income',
                    location='form')

parser.add_argument('NumberOfOpenCreditLinesAndLoans',
                    type=int,
                    required=True,
                    help='Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)',
                    location='form')

parser.add_argument('NumberOfTimes90DaysLate',
                    type=int,
                    required=True,
                    help='Number of times borrower has been 90 days or more past due.',
                    location='form')

parser.add_argument('NumberRealEstateLoansOrLines',
                    type=int,
                    required=True,
                    help='Number of mortgage and real estate loans including home equity lines of credit',
                    location='form')

parser.add_argument('NumberOfTime60-89DaysPastDueNotWorse',
                    type=int,
                    required=True,
                    help='Number of times borrower has been 60-89 days past due but no worse in the last 2 years.',
                    location='form')


parser.add_argument('NumberOfDependents',
                    type=int,
                    required=True,
                    help='Number of dependents in family excluding themselves (spouse, children etc.)',
                    location='form')

resource_fields = api.model('Resource', {'result': fields.String,})

@mt.route('/')
class MtBank(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def post(self):
        args = parser.parse_args()
        result = self.get_result(args)
        return result, 201

    def get_result(self, args):
        debtRatio = args["DebtRatio"]
        monthlyIncome = args["MonthlyIncome"]
        dependents = args["NumberOfDependents"]
        openCreditLinesAndLoans = args["NumberOfOpenCreditLinesAndLoans"]
        pastDue30Days = args["NumberOfTime30-59DaysPastDueNotWorse"]
        pastDue60Days = args["NumberOfTime60-89DaysPastDueNotWorse"]
        pastDue90Days = args["NumberOfTimes90DaysLate"]
        realEstateLoansOrLines = args["NumberRealEstateLoansOrLines"]
        unsecuredLines = args["RevolvingUtilizationOfUnsecuredLines"]
        age = args["Age"]


        X = np.array([[unsecuredLines, age, pastDue30Days,
                    debtRatio, monthlyIncome, openCreditLinesAndLoans,
                    pastDue90Days, realEstateLoansOrLines,
                    pastDue60Days, dependents]])

        clf = joblib.load('classifier.pkl')
        result = clf.predict(X)

        if result[0] == 1.0:
            result = "Denied"
        else:
            result = "Approved"

        return {"result": result}

if __name__ == '__main__':
    app.run(debug=False)