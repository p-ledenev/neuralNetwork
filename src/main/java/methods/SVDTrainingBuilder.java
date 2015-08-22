package methods;

import networks.*;
import org.encog.ml.*;
import org.encog.ml.data.*;
import org.encog.neural.rbf.*;
import org.encog.neural.rbf.training.*;

/**
 * Created by ledenev.p on 18.08.2015.
 */

public class SVDTrainingBuilder extends TrainingBuilder {

    public SVDTrainingBuilder(NetworkBuilder networkBuilder) {
        super(networkBuilder);
    }

    @Override
    protected SVDTraining createTraining(BasicML network, MLDataSet trainingSet) {
        return new SVDTraining((RBFNetwork) network, trainingSet);
    }
}
