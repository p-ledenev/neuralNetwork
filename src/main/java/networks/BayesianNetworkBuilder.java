package networks;

import org.encog.ml.*;
import org.encog.ml.bayesian.*;
import org.encog.ml.data.*;

/**
 * Created by ledenev.p on 20.08.2015.
 */
public class BayesianNetworkBuilder extends NetworkBuilder {

    @Override
    protected String getName() {
        return "BayesianNetwork";
    }

    @Override
    protected BasicML createNetwork(MLDataSet trainingSet) {
        return new BayesianNetwork();
    }
}
