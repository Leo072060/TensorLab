// #include <cmath>
// #include <iostream>
// #include <stdexcept>

// #include "kits/managed.hpp"
// #include "mat/mat.hpp"

// using namespace std;

// template <typename T = double> class RegressionEvaluation : public ManagedClass
// {
//   public:
//     RegressionEvaluation():isFitted(this->administrator)
//     RegressionEvaluation(const Mat<T> &pred_y, const Mat<T> &target_y);

//   public:
//     void fit(const Mat<T> &pred_y, const Mat<T> &target_y)
//     {
//         const_cast<const RegressionEvaluation *>(this)->fit(pred_y, target_y);
//     }
//     void report() const;
//     T    mean_absolute_error() const;
//     T    mean_squared_error() const;
//     T    root_mean_squared_error() const;
//     T    mean_absolute_percentage_error() const;
//     T    r2_score() const;

//   private:
//     void fit(const Mat<T> &pred_y, const Mat<T> &target_y) const;

//   private:
//     mutable bool               isFitted = false;
//     mutable ManagedVal<Mat<T>> TARGET_Y;
//     mutable ManagedVal<Mat<T>> PRED_Y;
//     mutable ManagedVal<Mat<T>> target_y_MINUS_pred_y;
//     mutable ManagedVal<Mat<T>> target_y_MINUS_mean_target_y;
//     mutable ManagedVal<T>      MAE;
//     mutable ManagedVal<T>      MSE;
//     mutable ManagedVal<T>      RMSE;
//     mutable ManagedVal<T>      MAPE;
//     mutable ManagedVal<T>      R2;
// };

// #pragma region lifecycle management
// template <typename T>
// RegressionEvaluation<T>::RegressionEvaluation()
//     : ManagedClass(), TARGET_Y(this->administrator), PRED_Y(this->administrator),
//       target_y_MINUS_pred_y(this->administrator), target_y_MINUS_mean_target_y(this->administrator),
//       MAE(this->administrator), MSE(this->administrator), RMSE(this->administrator), MAPE(this->administrator),
//       R2(this->administrator)
// {
// }
// template <typename T>
// RegressionEvaluation<T>::RegressionEvaluation(const Mat<T> &pred_y, const Mat<T> &target_y) : RegressionEvaluation()
// {
//     fit(pred_y, target_y);
// }
// #pragma endregion

// #pragma region function defination

// #pragma region member      functions
// template <typename T> void RegressionEvaluation<T>::report() const
// {
//     if (!isFitted) throw runtime_error("Error: The model must be fitted before generating the report.");
//     cout << "\t------- Regression Model Performance Report -------\n";
//     cout << "mean absolute error           "
//          << "\t" << mean_absolute_error() << endl;
//     cout << "mean squared error            "
//          << "\t" << mean_squared_error() << endl;
//     cout << "root mean squared error       "
//          << "\t" << root_mean_squared_error() << endl;
//     cout << "mean absolute percentage error"
//          << "\t" << mean_absolute_percentage_error() << endl;
//     cout << "r2 score                      "
//          << "\t" << r2_score() << endl;
// }
// template <typename T> T RegressionEvaluation<T>::mean_absolute_error() const
// {
//     if (MAE.readable()) return MAE.read();
//     if (!isFitted) throw runtime_error("Error: The model must be fitted before generating mean absolute error.");
//     this->record(MAE, mean(abs(target_y_MINUS_pred_y.read())));
//     return MAE.read();
// }
// template <typename T> T RegressionEvaluation<T>::mean_squared_error() const
// {
//     if (MSE.readable()) return MSE.read();
//     if (!isFitted) throw runtime_error("Error: The model must be fitted before generating mean squared error.");
//     this->record(MSE, mean(power(target_y_MINUS_pred_y.read(), 2)));
//     return MSE.read();
// }
// template <typename T> T RegressionEvaluation<T>::root_mean_squared_error() const
// {
//     if (RMSE.readable()) return RMSE.read();
//     if (!isFitted) throw runtime_error("Error: The model must be fitted before generating root mean squared error.");
//     this->record(RMSE, pow(mean_squared_error(), 0.5));
//     return RMSE.read();
// }
// template <typename T> T RegressionEvaluation<T>::mean_absolute_percentage_error() const
// {
//     if (MAPE.readable()) return MAPE.read();
//     if (!isFitted)
//         throw runtime_error("Error: The model must be fitted before generating mean absolute percentage error.");
//     this->record(MAPE, mean(abs(target_y_MINUS_pred_y.read() / TARGET_Y)));
//     return MAPE.read();
// }
// template <typename T> T RegressionEvaluation<T>::r2_score() const
// {
//     if (R2.readable()) return R2.read();
//     if (!isFitted) throw runtime_error("Error: The model must be fitted before generating r2 score.");
//     this->record(R2,
//                  1 - sum(power(target_y_MINUS_pred_y.read(), 2)) / sum(power(target_y_MINUS_mean_target_y.read(), 2)));
//     return R2.read();
// }
// template <typename T> void RegressionEvaluation<T>::fit(const Mat<T> &pred_y, const Mat<T> &target_y) const
// {
//     if (pred_y.size_column() != 1) throw invalid_argument("Error: Matrix pred_y must be single-column matrix.");
//     if (target_y.size_column() != 1) throw invalid_argument("Error: Matrix target_y must be single-column matrix.");
//     this->refresh();
//     this->record(TARGET_Y, target_y);
//     this->record(PRED_Y, pred_y);
//     // calculate
//     Mat<T> tmp(target_y.size_row(), 1);
//     // calculate target_y_MINUS_pred_y
//     for (size_t i = 0; i < target_y.size_row(); ++i)
//         tmp.iloc(i, 0) = target_y.iloc(i, 0) - pred_y.iloc(i, 0);
//     this->record(target_y_MINUS_pred_y, tmp);
//     // calculate target_y_MINUS_mean_target_y
//     for (size_t i = 0; i < target_y.size_row(); ++i)
//         tmp.iloc(i, 0) = target_y.iloc(i, 0) - mean(TARGET_Y.read());
//     this->record(target_y_MINUS_mean_target_y, tmp);
//     isFitted = true;
// }
// #pragma endregion

// #pragma endregion

// template <typename T> class ClassificationEvaluation : public ManagedClass
// {
//   public:
//     ClassificationEvaluation();
//     ClassificationEvaluation(const Mat<string> &pred_y, const Mat<string> &target_y);

//     // * * * * * * * functions * * * * * * *
//   public:
//     void fit(const Mat<string> &pred_y, const Mat<string> &target_y)
//     {
//         const_cast<const ClassificationEvaluation<T> *>(this)->fit(pred_y, target_y);
//     }
//     void        report() const;
//     Mat<size_t> confusionMatrix() const;
//     T           accuracy() const;
//     T           error_rate() const;
//     Mat<T>      percision() const;
//     Mat<T>      recall() const;

//   private:
//     void fit(const Mat<string> &pred_y, const Mat<string> &target_y) const;
//     // * * * * * * * attributes * * * * * * *
//   private:
//     mutable bool                    isFitted = false;
//     mutable ManagedVal<Mat<string>> TARGET_Y;
//     mutable ManagedVal<Mat<string>> PRED_Y;
//     mutable ManagedVal<Mat<size_t>> CONFUSION_MATRIX;
//     mutable ManagedVal<T>           ACCURACY;
//     mutable ManagedVal<T>           ERROR_RATE;
//     mutable ManagedVal<Mat<T>>      PERCISION;
//     mutable ManagedVal<Mat<T>>      RECALL;
// };

// #pragma region lifecycle management
// template <typename T>
// ClassificationEvaluation<T>::ClassificationEvaluation()
//     : ManagedClass(), TARGET_Y(this->administrator), PRED_Y(this->administrator), CONFUSION_MATRIX(this->administrator),
//       ACCURACY(this->administrator), ERROR_RATE(this->administrator), PERCISION(this->administrator),
//       RECALL(this->administrator)
// {
// }
// template <typename T>
// ClassificationEvaluation<T>::ClassificationEvaluation(const Mat<string> &pred_y, const Mat<string> &target_y)
//     : ClassificationEvaluation()
// {
//     fit(pred_y, target_y);
// }
// #pragma endregion

// #pragma region function definition

// #pragma region member      functions
// template <typename T> void ClassificationEvaluation<T>::report() const
// {
//     if (!isFitted) throw runtime_error("Error: The model must be fitted before generating the report.");
//     cout << "\t------- Classification Model Performance Report -------\n";
//     cout << "confusion matrix : " << endl;
//     display(CONFUSION_MATRIX.read(), WITH_NAME);
//     cout << "accuracy                      "
//          << "\t" << accuracy() << endl;
//     cout << "error rate                    "
//          << "\t" << error_rate() << endl;
//     cout << "percision : " << endl;
//     display(percision(), WITH_NAME);
//     cout << "recall : " << endl;
//     display(recall(), WITH_NAME);
// }
// template <typename T> Mat<size_t> ClassificationEvaluation<T>::confusionMatrix() const
// {
//     if (!isFitted) throw runtime_error("Error: The model must be fitted before generating confusion matrix.");
//     Mat<string> types_target_y = unique(TARGET_Y.read());
//     Mat<string> types_pred_y   = unique(PRED_Y.read());
//     Mat<string> types_y        = unique((types_target_y, types_pred_y));
//     if (types_y.size() > types_target_y.size())
//         throw logic_error("Error: Predicted labels contain classes not present in the target labels.");

//     types_target_y.sort_column(0);
//     Mat<size_t> confusionMat(types_target_y.size_column(), types_target_y.size_column());
//     for (size_t i = 0; i < TARGET_Y.read().size_row(); ++i)
//     {
//         confusionMat.iloc(types_target_y.find(TARGET_Y.read().iloc(i, 0)).iloc(0, 1),
//                           types_target_y.find(PRED_Y.read().iloc(i, 0)).iloc(0, 1)) += 1;
//     }
//     for (size_t i = 0; i < confusionMat.size_row(); ++i)
//     {
//         confusionMat.iloc_rowName(i) = types_target_y.iloc(0, i);
//         confusionMat.iloc_colName(i) = types_target_y.iloc(0, i);
//     }
//     this->record(CONFUSION_MATRIX, confusionMat);
//     return CONFUSION_MATRIX.read();
// }
// template <typename T> T ClassificationEvaluation<T>::accuracy() const
// {
//     if (!isFitted) throw runtime_error("Error: The model must be fitted before generating accuracy.");
//     if (ACCURACY.readable()) return ACCURACY.read();

//     T sum_TFPN  = sum(CONFUSION_MATRIX.read());
//     T sum_TP_TN = 0;
//     for (size_t i = 0; i < CONFUSION_MATRIX.read().size_row(); ++i)
//         sum_TP_TN += CONFUSION_MATRIX.read().iloc(i, i);
//     this->record(ACCURACY, sum_TP_TN / sum_TFPN);
//     return ACCURACY.read();
// }
// template <typename T> T ClassificationEvaluation<T>::error_rate() const
// {
//     if (!isFitted) throw runtime_error("Error: The model must be fitted before generating error rate.");
//     if (ERROR_RATE.readable()) return ERROR_RATE.read();
//     this->record(ERROR_RATE, 1 - accuracy());
//     return ERROR_RATE.read();
// }
// template <typename T> Mat<T> ClassificationEvaluation<T>::percision() const
// {
//     if (!isFitted) throw runtime_error("Error: The model must be fitted before generating percision.");
//     if (PERCISION.readable()) return PERCISION.read();

//     Mat<size_t> sum_TP_FP = sum_row(CONFUSION_MATRIX.read());
//     Mat<T>      ret(1, CONFUSION_MATRIX.read().size_column());
//     for (size_t i = 0; i < ret.size_column(); ++i)
//     {
//         ret.iloc(0, i)      = CONFUSION_MATRIX.read().iloc(i, i) * 1.0 / sum_TP_FP.iloc(i, 0);
//         ret.iloc_colName(i) = CONFUSION_MATRIX.read().iloc_colName(i);
//     }
//     ret.iloc_rowName(0) = "percision";
//     this->record(PERCISION, ret);

//     return PERCISION.read();
// }
// template <typename T> Mat<T> ClassificationEvaluation<T>::recall() const
// {
//     if (!isFitted) throw runtime_error("Error: The model must be fitted before generating recall.");
//     if (RECALL.readable()) return RECALL.read();

//     Mat<size_t> sum_TP_FP = sum_column(CONFUSION_MATRIX.read());
//     Mat<T>      ret(1, CONFUSION_MATRIX.read().size_column());
//     for (size_t i = 0; i < ret.size_column(); ++i)
//     {
//         ret.iloc(0, i)      = CONFUSION_MATRIX.read().iloc(i, i) * 1.0 / sum_TP_FP.iloc(0, i);
//         ret.iloc_colName(i) = CONFUSION_MATRIX.read().iloc_colName(i);
//     }
//     ret.iloc_rowName(0) = "recall";
//     this->record(RECALL, ret);
//     return RECALL.read();
// }
// template <typename T>
// void ClassificationEvaluation<T>::fit(const Mat<string> &pred_y, const Mat<string> &target_y) const
// {
//     if (pred_y.size_column() != 1) throw invalid_argument("Error: Matrix pred_y must be single-column matrix.");
//     if (target_y.size_column() != 1) throw invalid_argument("Error: Matrix target_y must be single-column matrix.");
//     this->refresh();
//     this->record(TARGET_Y, target_y);
//     this->record(PRED_Y, pred_y);
//     Mat<string> types_pred_y   = unique(pred_y);
//     Mat<string> types_target_y = unique(target_y);
//     isFitted                   = true;
//     confusionMatrix();
// }
// #pragma endregion

// #pragma endregion