package methods;

import networks.*;
import org.encog.ml.*;
import org.encog.ml.data.*;
import org.encog.ml.train.*;
import org.encog.neural.networks.*;
import org.encog.neural.networks.training.nm.*;

/**
 * Created by ledenev.p on 18.08.2015.
 */
public class NelderMeadTrainingBuilder extends TrainingBuilder {

    public NelderMeadTrainingBuilder(NetworkBuilder networkBuilder) {
        super(networkBuilder);
    }

    @Override
    protected BasicTraining createTraining(BasicML network, MLDataSet trainingSet) {
        return new NelderMeadTraining((BasicNetwork) network, trainingSet);
    }
}
