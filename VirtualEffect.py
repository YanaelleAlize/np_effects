class VirtualEffect :
    """
    Virtual Class for numpy Effects that activates on (an / a / multiple) (iterable(s) / numpy array(s))
    """

    def __init__(self) -> None :
        """
        Init an instance of an Effect as a super Class for heir effects
        :key meta: dict storing information about the effect to be applied
        :key meta.__desc__: stores a description of the Effect
        :key meta.__len_op__: stores the number of parameter of the effect,
        :key meta.__len__: size of the objects (images GrayScale 1, images RGB 3)
        :return: None
        """
        self.meta = {
            "__desc__" : "Virtual Class for numpy Effects that activates on (an / a / multiple) (iterable(s) / numpy array(s))",
            "__len_op__" :0,
            "__len__" : 0
            }

    def __len__(self) -> int :
        return self.meta["__len__"]

    def __repr__(self) -> str :
        raise NotImplementedError

    def apply(self, map_able_objects : list = [], alpha : float = 1) -> None:
        """
        Apply the Operator to Objects
        :param map_able_objects: objects corresponding to the number of parameter of the operator (binary for 2),
        on which the effect will be applied
        :param alpha: percentage of application of the Effect
        :return: None, Transforms map_able_objects
        """
        assert len(map_able_objects) == self.meta["__len_op__"], "Operator must apply on dimension objects of same dimension"
        raise NotImplementedError

    def precompute(self) -> None:
        """
        Precomputes the Operator to be applyable to objects
        :return: None, Will store in self all necessary data for the operator top be functional
        """
        raise NotImplementedError
