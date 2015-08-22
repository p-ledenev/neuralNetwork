package methods;

import networks.*;
import org.encog.ml.*;
import org.encog.ml.data.*;
import org.encog.ml.svm.*;
import org.encog.ml.svm.training.*;
import org.encog.ml.train.*;

/**
 * Created by ledenev.p on 20.08.2015.
 */
public class SVMSearchTrainingBuilder extends TrainingBuilder {

    public SVMSearchTrainingBuilder(NetworkBuilder networkBuilder) {
        super(networkBuilder);
    }

    @Override
    protected BasicTraining createTraining(BasicML network, MLDataSet trainingSet) {
        return new SVMSearchTrain((SVM) network, trainingSet);
    }
}
