package networks;

import org.encog.ml.*;
import org.encog.ml.data.*;
import org.encog.ml.svm.*;

/**
 * Created by ledenev.p on 20.08.2015.
 */
public class SVMBuilder extends NetworkBuilder {
    @Override
    protected String getName() {
        return "SVM";
    }

    @Override
    protected BasicML createNetwork(MLDataSet trainingSet) {
        return new SVM(trainingSet.getInputSize(), SVMType.SupportVectorOneClass, KernelType.RadialBasisFunction);
    }
}
