package methods;

import networks.*;
import org.encog.ml.*;
import org.encog.ml.data.*;
import org.encog.ml.train.*;
import org.encog.neural.networks.*;
import org.encog.neural.networks.training.propagation.scg.*;

/**
 * Created by ledenev.p on 20.08.2015.
 */
public class ScaledConjugateGradientTrainingBuilder extends TrainingBuilder {

    public ScaledConjugateGradientTrainingBuilder(NetworkBuilder networkBuilder) {
        super(networkBuilder);
    }

    @Override
    protected BasicTraining createTraining(BasicML network, MLDataSet trainingSet) {
        return new ScaledConjugateGradient((BasicNetwork) network, trainingSet);
    }
}
