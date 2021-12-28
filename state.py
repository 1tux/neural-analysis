class get_state(object):
   def __new__(cls):

       if not hasattr(cls, 'instance'):

           cls.instance = super(get_state, cls).__new__(cls)

       return cls.instance

get_state().train_test_ratio = 0.5