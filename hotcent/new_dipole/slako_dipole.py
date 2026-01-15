import numpy as np
np.set_printoptions(precision=16)

"""nonvanishing SlaKo integrals over phi for dipole elements named in the form
Y1*Y1*Y2
sk_label: tuple
   (idx, l, m, l, m, l, m) with idx:unique number of integral according to increasing quantum numbers 
   l,m: quantum numbers of the respective harmonics 
"""
INTEGRALS_DIPOLE = {
	(1, 0, 0, 1, -1, 1, -1): lambda c1, c2, s1, s2: 0.375*s1*s2/np.sqrt(np.pi),
	(5, 0, 0, 1, -1, 2, -1): lambda c1, c2, s1, s2: 0.375*np.sqrt(5)*s1*s2*c2/np.sqrt(np.pi),
	(11, 0, 0, 1, -1, 3, -1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(14)*(5*c2**2 - 1)*s1*s2/np.sqrt(np.pi),
	(16, 0, 0, 1, 0, 0, 0): lambda c1, c2, s1, s2: 0.25*np.sqrt(3)*c1/np.sqrt(np.pi),
	(18, 0, 0, 1, 0, 1, 0): lambda c1, c2, s1, s2: 0.75*c1*c2/np.sqrt(np.pi),
	(22, 0, 0, 1, 0, 2, 0): lambda c1, c2, s1, s2: 0.125*np.sqrt(15)*(3*c2**2 - 1)*c1/np.sqrt(np.pi),
	(28, 0, 0, 1, 0, 3, 0): lambda c1, c2, s1, s2: 0.125*np.sqrt(21)*(5*c2**3 - 3*c2)*c1/np.sqrt(np.pi),
	(35, 0, 0, 1, 1, 1, 1): lambda c1, c2, s1, s2: 0.375*s1*s2/np.sqrt(np.pi),
	(39, 0, 0, 1, 1, 2, 1): lambda c1, c2, s1, s2: 0.375*np.sqrt(5)*s1*s2*c2/np.sqrt(np.pi),
	(45, 0, 0, 1, 1, 3, 1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(14)*(5*c2**2 - 1)*s1*s2/np.sqrt(np.pi),
	(48, 1, -1, 1, -1, 0, 0): lambda c1, c2, s1, s2: 0.375*s1**2/np.sqrt(np.pi),
	(50, 1, -1, 1, -1, 1, 0): lambda c1, c2, s1, s2: 0.375*np.sqrt(3)*s1**2*c2/np.sqrt(np.pi),
	(54, 1, -1, 1, -1, 2, 0): lambda c1, c2, s1, s2: 0.1875*np.sqrt(5)*(3*c2**2 - 1)*s1**2/np.sqrt(np.pi),
	(56, 1, -1, 1, -1, 2, 2): lambda c1, c2, s1, s2: -0.09375*np.sqrt(15)*s1**2*s2**2/np.sqrt(np.pi),
	(60, 1, -1, 1, -1, 3, 0): lambda c1, c2, s1, s2: 0.1875*np.sqrt(7)*(5*c2**3 - 3*c2)*s1**2/np.sqrt(np.pi),
	(62, 1, -1, 1, -1, 3, 2): lambda c1, c2, s1, s2: -0.09375*np.sqrt(105)*s1**2*s2**2*c2/np.sqrt(np.pi),
	(65, 1, -1, 1, 0, 1, -1): lambda c1, c2, s1, s2: 0.375*np.sqrt(3)*s1*s2*c1/np.sqrt(np.pi),
	(69, 1, -1, 1, 0, 2, -1): lambda c1, c2, s1, s2: 0.375*np.sqrt(15)*s1*s2*c1*c2/np.sqrt(np.pi),
	(75, 1, -1, 1, 0, 3, -1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(42)*(5*c2**2 - 1)*s1*s2*c1/np.sqrt(np.pi),
	(84, 1, -1, 1, 1, 2, -2): lambda c1, c2, s1, s2: 0.09375*np.sqrt(15)*s1**2*s2**2/np.sqrt(np.pi),
	(90, 1, -1, 1, 1, 3, -2): lambda c1, c2, s1, s2: 0.09375*np.sqrt(105)*s1**2*s2**2*c2/np.sqrt(np.pi),
	(97, 1, 0, 1, -1, 1, -1): lambda c1, c2, s1, s2: 0.375*np.sqrt(3)*s1*s2*c1/np.sqrt(np.pi),
	(101, 1, 0, 1, -1, 2, -1): lambda c1, c2, s1, s2: 0.375*np.sqrt(15)*s1*s2*c1*c2/np.sqrt(np.pi),
	(107, 1, 0, 1, -1, 3, -1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(42)*(5*c2**2 - 1)*s1*s2*c1/np.sqrt(np.pi),
	(112, 1, 0, 1, 0, 0, 0): lambda c1, c2, s1, s2: 0.75*c1**2/np.sqrt(np.pi),
	(114, 1, 0, 1, 0, 1, 0): lambda c1, c2, s1, s2: 0.75*np.sqrt(3)*c1**2*c2/np.sqrt(np.pi),
	(118, 1, 0, 1, 0, 2, 0): lambda c1, c2, s1, s2: 0.375*np.sqrt(5)*(3*c2**2 - 1)*c1**2/np.sqrt(np.pi),
	(124, 1, 0, 1, 0, 3, 0): lambda c1, c2, s1, s2: 0.375*np.sqrt(7)*(5*c2**3 - 3*c2)*c1**2/np.sqrt(np.pi),
	(131, 1, 0, 1, 1, 1, 1): lambda c1, c2, s1, s2: 0.375*np.sqrt(3)*s1*s2*c1/np.sqrt(np.pi),
	(135, 1, 0, 1, 1, 2, 1): lambda c1, c2, s1, s2: 0.375*np.sqrt(15)*s1*s2*c1*c2/np.sqrt(np.pi),
	(141, 1, 0, 1, 1, 3, 1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(42)*(5*c2**2 - 1)*s1*s2*c1/np.sqrt(np.pi),
	(148, 1, 1, 1, -1, 2, -2): lambda c1, c2, s1, s2: 0.09375*np.sqrt(15)*s1**2*s2**2/np.sqrt(np.pi),
	(154, 1, 1, 1, -1, 3, -2): lambda c1, c2, s1, s2: 0.09375*np.sqrt(105)*s1**2*s2**2*c2/np.sqrt(np.pi),
	(163, 1, 1, 1, 0, 1, 1): lambda c1, c2, s1, s2: 0.375*np.sqrt(3)*s1*s2*c1/np.sqrt(np.pi),
	(167, 1, 1, 1, 0, 2, 1): lambda c1, c2, s1, s2: 0.375*np.sqrt(15)*s1*s2*c1*c2/np.sqrt(np.pi),
	(173, 1, 1, 1, 0, 3, 1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(42)*(5*c2**2 - 1)*s1*s2*c1/np.sqrt(np.pi),
	(176, 1, 1, 1, 1, 0, 0): lambda c1, c2, s1, s2: 0.375*s1**2/np.sqrt(np.pi),
	(178, 1, 1, 1, 1, 1, 0): lambda c1, c2, s1, s2: 0.375*np.sqrt(3)*s1**2*c2/np.sqrt(np.pi),
	(182, 1, 1, 1, 1, 2, 0): lambda c1, c2, s1, s2: 0.1875*np.sqrt(5)*(3*c2**2 - 1)*s1**2/np.sqrt(np.pi),
	(184, 1, 1, 1, 1, 2, 2): lambda c1, c2, s1, s2: 0.09375*np.sqrt(15)*s1**2*s2**2/np.sqrt(np.pi),
	(188, 1, 1, 1, 1, 3, 0): lambda c1, c2, s1, s2: 0.1875*np.sqrt(7)*(5*c2**3 - 3*c2)*s1**2/np.sqrt(np.pi),
	(190, 1, 1, 1, 1, 3, 2): lambda c1, c2, s1, s2: 0.09375*np.sqrt(105)*s1**2*s2**2*c2/np.sqrt(np.pi),
	(195, 2, -2, 1, -1, 1, 1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(15)*s1**3*s2/np.sqrt(np.pi),
	(199, 2, -2, 1, -1, 2, 1): lambda c1, c2, s1, s2: 0.46875*np.sqrt(3)*s1**3*s2*c2/np.sqrt(np.pi),
	(205, 2, -2, 1, -1, 3, 1): lambda c1, c2, s1, s2: 0.0234375*np.sqrt(210)*(5*c2**2 - 1)*s1**3*s2/np.sqrt(np.pi),
	(207, 2, -2, 1, -1, 3, 3): lambda c1, c2, s1, s2: -0.1171875*np.sqrt(14)*s1**3*s2**3/np.sqrt(np.pi),
	(212, 2, -2, 1, 0, 2, -2): lambda c1, c2, s1, s2: 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi),
	(218, 2, -2, 1, 0, 3, -2): lambda c1, c2, s1, s2: 0.46875*np.sqrt(21)*s1**2*s2**2*c1*c2/np.sqrt(np.pi),
	(225, 2, -2, 1, 1, 1, -1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(15)*s1**3*s2/np.sqrt(np.pi),
	(229, 2, -2, 1, 1, 2, -1): lambda c1, c2, s1, s2: 0.46875*np.sqrt(3)*s1**3*s2*c2/np.sqrt(np.pi),
	(233, 2, -2, 1, 1, 3, -3): lambda c1, c2, s1, s2: 0.1171875*np.sqrt(14)*s1**3*s2**3/np.sqrt(np.pi),
	(235, 2, -2, 1, 1, 3, -1): lambda c1, c2, s1, s2: 0.0234375*np.sqrt(210)*(5*c2**2 - 1)*s1**3*s2/np.sqrt(np.pi),
	(240, 2, -1, 1, -1, 0, 0): lambda c1, c2, s1, s2: 0.375*np.sqrt(5)*s1**2*c1/np.sqrt(np.pi),
	(242, 2, -1, 1, -1, 1, 0): lambda c1, c2, s1, s2: 0.375*np.sqrt(15)*s1**2*c1*c2/np.sqrt(np.pi),
	(246, 2, -1, 1, -1, 2, 0): lambda c1, c2, s1, s2: 0.9375*(3*c2**2 - 1)*s1**2*c1/np.sqrt(np.pi),
	(248, 2, -1, 1, -1, 2, 2): lambda c1, c2, s1, s2: -0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi),
	(252, 2, -1, 1, -1, 3, 0): lambda c1, c2, s1, s2: 0.1875*np.sqrt(35)*(5*c2**3 - 3*c2)*s1**2*c1/np.sqrt(np.pi),
	(254, 2, -1, 1, -1, 3, 2): lambda c1, c2, s1, s2: -0.46875*np.sqrt(21)*s1**2*s2**2*c1*c2/np.sqrt(np.pi),
	(257, 2, -1, 1, 0, 1, -1): lambda c1, c2, s1, s2: 0.375*np.sqrt(15)*s1*s2*c1**2/np.sqrt(np.pi),
	(261, 2, -1, 1, 0, 2, -1): lambda c1, c2, s1, s2: 1.875*np.sqrt(3)*s1*s2*c1**2*c2/np.sqrt(np.pi),
	(267, 2, -1, 1, 0, 3, -1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(210)*(5*c2**2 - 1)*s1*s2*c1**2/np.sqrt(np.pi),
	(276, 2, -1, 1, 1, 2, -2): lambda c1, c2, s1, s2: 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi),
	(282, 2, -1, 1, 1, 3, -2): lambda c1, c2, s1, s2: 0.46875*np.sqrt(21)*s1**2*s2**2*c1*c2/np.sqrt(np.pi),
	(289, 2, 0, 1, -1, 1, -1): lambda c1, c2, s1, s2: 0.1875*np.sqrt(5)*(3*c1**2 - 1)*s1*s2/np.sqrt(np.pi),
	(293, 2, 0, 1, -1, 2, -1): lambda c1, c2, s1, s2: 0.9375*(3*c1**2 - 1)*s1*s2*c2/np.sqrt(np.pi),
	(299, 2, 0, 1, -1, 3, -1): lambda c1, c2, s1, s2: 0.046875*np.sqrt(70)*(3*c1**2 - 1)*(5*c2**2 - 1)*s1*s2/np.sqrt(np.pi),
	(304, 2, 0, 1, 0, 0, 0): lambda c1, c2, s1, s2: 0.125*np.sqrt(15)*(3*c1**2 - 1)*c1/np.sqrt(np.pi),
	(306, 2, 0, 1, 0, 1, 0): lambda c1, c2, s1, s2: 0.375*np.sqrt(5)*(3*c1**2 - 1)*c1*c2/np.sqrt(np.pi),
	(310, 2, 0, 1, 0, 2, 0): lambda c1, c2, s1, s2: 0.3125*np.sqrt(3)*(3*c1**2 - 1)*(3*c2**2 - 1)*c1/np.sqrt(np.pi),
	(316, 2, 0, 1, 0, 3, 0): lambda c1, c2, s1, s2: 0.0625*np.sqrt(105)*(3*c1**2 - 1)*(5*c2**3 - 3*c2)*c1/np.sqrt(np.pi),
	(323, 2, 0, 1, 1, 1, 1): lambda c1, c2, s1, s2: 0.1875*np.sqrt(5)*(3*c1**2 - 1)*s1*s2/np.sqrt(np.pi),
	(327, 2, 0, 1, 1, 2, 1): lambda c1, c2, s1, s2: 0.9375*(3*c1**2 - 1)*s1*s2*c2/np.sqrt(np.pi),
	(333, 2, 0, 1, 1, 3, 1): lambda c1, c2, s1, s2: 0.046875*np.sqrt(70)*(3*c1**2 - 1)*(5*c2**2 - 1)*s1*s2/np.sqrt(np.pi),
	(340, 2, 1, 1, -1, 2, -2): lambda c1, c2, s1, s2: 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi),
	(346, 2, 1, 1, -1, 3, -2): lambda c1, c2, s1, s2: 0.46875*np.sqrt(21)*s1**2*s2**2*c1*c2/np.sqrt(np.pi),
	(355, 2, 1, 1, 0, 1, 1): lambda c1, c2, s1, s2: 0.375*np.sqrt(15)*s1*s2*c1**2/np.sqrt(np.pi),
	(359, 2, 1, 1, 0, 2, 1): lambda c1, c2, s1, s2: 1.875*np.sqrt(3)*s1*s2*c1**2*c2/np.sqrt(np.pi),
	(365, 2, 1, 1, 0, 3, 1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(210)*(5*c2**2 - 1)*s1*s2*c1**2/np.sqrt(np.pi),
	(368, 2, 1, 1, 1, 0, 0): lambda c1, c2, s1, s2: 0.375*np.sqrt(5)*s1**2*c1/np.sqrt(np.pi),
	(370, 2, 1, 1, 1, 1, 0): lambda c1, c2, s1, s2: 0.375*np.sqrt(15)*s1**2*c1*c2/np.sqrt(np.pi),
	(374, 2, 1, 1, 1, 2, 0): lambda c1, c2, s1, s2: 0.9375*(3*c2**2 - 1)*s1**2*c1/np.sqrt(np.pi),
	(376, 2, 1, 1, 1, 2, 2): lambda c1, c2, s1, s2: 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi),
	(380, 2, 1, 1, 1, 3, 0): lambda c1, c2, s1, s2: 0.1875*np.sqrt(35)*(5*c2**3 - 3*c2)*s1**2*c1/np.sqrt(np.pi),
	(382, 2, 1, 1, 1, 3, 2): lambda c1, c2, s1, s2: 0.46875*np.sqrt(21)*s1**2*s2**2*c1*c2/np.sqrt(np.pi),
	(385, 2, 2, 1, -1, 1, -1): lambda c1, c2, s1, s2: -0.09375*np.sqrt(15)*s1**3*s2/np.sqrt(np.pi),
	(389, 2, 2, 1, -1, 2, -1): lambda c1, c2, s1, s2: -0.46875*np.sqrt(3)*s1**3*s2*c2/np.sqrt(np.pi),
	(393, 2, 2, 1, -1, 3, -3): lambda c1, c2, s1, s2: 0.1171875*np.sqrt(14)*s1**3*s2**3/np.sqrt(np.pi),
	(395, 2, 2, 1, -1, 3, -1): lambda c1, c2, s1, s2: -0.0234375*np.sqrt(210)*(5*c2**2 - 1)*s1**3*s2/np.sqrt(np.pi),
	(408, 2, 2, 1, 0, 2, 2): lambda c1, c2, s1, s2: 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi),
	(414, 2, 2, 1, 0, 3, 2): lambda c1, c2, s1, s2: 0.46875*np.sqrt(21)*s1**2*s2**2*c1*c2/np.sqrt(np.pi),
	(419, 2, 2, 1, 1, 1, 1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(15)*s1**3*s2/np.sqrt(np.pi),
	(423, 2, 2, 1, 1, 2, 1): lambda c1, c2, s1, s2: 0.46875*np.sqrt(3)*s1**3*s2*c2/np.sqrt(np.pi),
	(429, 2, 2, 1, 1, 3, 1): lambda c1, c2, s1, s2: 0.0234375*np.sqrt(210)*(5*c2**2 - 1)*s1**3*s2/np.sqrt(np.pi),
	(431, 2, 2, 1, 1, 3, 3): lambda c1, c2, s1, s2: 0.1171875*np.sqrt(14)*s1**3*s2**3/np.sqrt(np.pi),
	(440, 3, -3, 1, -1, 2, 2): lambda c1, c2, s1, s2: 0.1171875*np.sqrt(14)*s1**4*s2**2/np.sqrt(np.pi),
	(446, 3, -3, 1, -1, 3, 2): lambda c1, c2, s1, s2: 0.8203125*np.sqrt(2)*s1**4*s2**2*c2/np.sqrt(np.pi),
	(457, 3, -3, 1, 0, 3, -3): lambda c1, c2, s1, s2: 0.546875*np.sqrt(3)*s1**3*s2**3*c1/np.sqrt(np.pi),
	(468, 3, -3, 1, 1, 2, -2): lambda c1, c2, s1, s2: 0.1171875*np.sqrt(14)*s1**4*s2**2/np.sqrt(np.pi),
	(474, 3, -3, 1, 1, 3, -2): lambda c1, c2, s1, s2: 0.8203125*np.sqrt(2)*s1**4*s2**2*c2/np.sqrt(np.pi),
	(483, 3, -2, 1, -1, 1, 1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(105)*s1**3*s2*c1/np.sqrt(np.pi),
	(487, 3, -2, 1, -1, 2, 1): lambda c1, c2, s1, s2: 0.46875*np.sqrt(21)*s1**3*s2*c1*c2/np.sqrt(np.pi),
	(493, 3, -2, 1, -1, 3, 1): lambda c1, c2, s1, s2: 0.1640625*np.sqrt(30)*(5*c2**2 - 1)*s1**3*s2*c1/np.sqrt(np.pi),
	(495, 3, -2, 1, -1, 3, 3): lambda c1, c2, s1, s2: -0.8203125*np.sqrt(2)*s1**3*s2**3*c1/np.sqrt(np.pi),
	(500, 3, -2, 1, 0, 2, -2): lambda c1, c2, s1, s2: 0.46875*np.sqrt(21)*s1**2*s2**2*c1**2/np.sqrt(np.pi),
	(506, 3, -2, 1, 0, 3, -2): lambda c1, c2, s1, s2: 3.28125*np.sqrt(3)*s1**2*s2**2*c1**2*c2/np.sqrt(np.pi),
	(513, 3, -2, 1, 1, 1, -1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(105)*s1**3*s2*c1/np.sqrt(np.pi),
	(517, 3, -2, 1, 1, 2, -1): lambda c1, c2, s1, s2: 0.46875*np.sqrt(21)*s1**3*s2*c1*c2/np.sqrt(np.pi),
	(521, 3, -2, 1, 1, 3, -3): lambda c1, c2, s1, s2: 0.8203125*np.sqrt(2)*s1**3*s2**3*c1/np.sqrt(np.pi),
	(523, 3, -2, 1, 1, 3, -1): lambda c1, c2, s1, s2: 0.1640625*np.sqrt(30)*(5*c2**2 - 1)*s1**3*s2*c1/np.sqrt(np.pi),
	(528, 3, -1, 1, -1, 0, 0): lambda c1, c2, s1, s2: 0.09375*np.sqrt(14)*(5*c1**2 - 1)*s1**2/np.sqrt(np.pi),
	(530, 3, -1, 1, -1, 1, 0): lambda c1, c2, s1, s2: 0.09375*np.sqrt(42)*(5*c1**2 - 1)*s1**2*c2/np.sqrt(np.pi),
	(534, 3, -1, 1, -1, 2, 0): lambda c1, c2, s1, s2: 0.046875*np.sqrt(70)*(5*c1**2 - 1)*(3*c2**2 - 1)*s1**2/np.sqrt(np.pi),
	(536, 3, -1, 1, -1, 2, 2): lambda c1, c2, s1, s2: -0.0234375*np.sqrt(210)*(5*c1**2 - 1)*s1**2*s2**2/np.sqrt(np.pi),
	(540, 3, -1, 1, -1, 3, 0): lambda c1, c2, s1, s2: 0.328125*np.sqrt(2)*(5*c1**2 - 1)*(5*c2**3 - 3*c2)*s1**2/np.sqrt(np.pi),
	(542, 3, -1, 1, -1, 3, 2): lambda c1, c2, s1, s2: -0.1640625*np.sqrt(30)*(5*c1**2 - 1)*s1**2*s2**2*c2/np.sqrt(np.pi),
	(545, 3, -1, 1, 0, 1, -1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(42)*(5*c1**2 - 1)*s1*s2*c1/np.sqrt(np.pi),
	(549, 3, -1, 1, 0, 2, -1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(210)*(5*c1**2 - 1)*s1*s2*c1*c2/np.sqrt(np.pi),
	(555, 3, -1, 1, 0, 3, -1): lambda c1, c2, s1, s2: 0.328125*np.sqrt(3)*(5*c1**2 - 1)*(5*c2**2 - 1)*s1*s2*c1/np.sqrt(np.pi),
	(564, 3, -1, 1, 1, 2, -2): lambda c1, c2, s1, s2: 0.0234375*np.sqrt(210)*(5*c1**2 - 1)*s1**2*s2**2/np.sqrt(np.pi),
	(570, 3, -1, 1, 1, 3, -2): lambda c1, c2, s1, s2: 0.1640625*np.sqrt(30)*(5*c1**2 - 1)*s1**2*s2**2*c2/np.sqrt(np.pi),
	(577, 3, 0, 1, -1, 1, -1): lambda c1, c2, s1, s2: 0.1875*np.sqrt(7)*(5*c1**3 - 3*c1)*s1*s2/np.sqrt(np.pi),
	(581, 3, 0, 1, -1, 2, -1): lambda c1, c2, s1, s2: 0.1875*np.sqrt(35)*(5*c1**3 - 3*c1)*s1*s2*c2/np.sqrt(np.pi),
	(587, 3, 0, 1, -1, 3, -1): lambda c1, c2, s1, s2: 0.328125*np.sqrt(2)*(5*c1**3 - 3*c1)*(5*c2**2 - 1)*s1*s2/np.sqrt(np.pi),
	(592, 3, 0, 1, 0, 0, 0): lambda c1, c2, s1, s2: 0.125*np.sqrt(21)*(5*c1**3 - 3*c1)*c1/np.sqrt(np.pi),
	(594, 3, 0, 1, 0, 1, 0): lambda c1, c2, s1, s2: 0.375*np.sqrt(7)*(5*c1**3 - 3*c1)*c1*c2/np.sqrt(np.pi),
	(598, 3, 0, 1, 0, 2, 0): lambda c1, c2, s1, s2: 0.0625*np.sqrt(105)*(5*c1**3 - 3*c1)*(3*c2**2 - 1)*c1/np.sqrt(np.pi),
	(604, 3, 0, 1, 0, 3, 0): lambda c1, c2, s1, s2: 0.4375*np.sqrt(3)*(5*c1**3 - 3*c1)*(5*c2**3 - 3*c2)*c1/np.sqrt(np.pi),
	(611, 3, 0, 1, 1, 1, 1): lambda c1, c2, s1, s2: 0.1875*np.sqrt(7)*(5*c1**3 - 3*c1)*s1*s2/np.sqrt(np.pi),
	(615, 3, 0, 1, 1, 2, 1): lambda c1, c2, s1, s2: 0.1875*np.sqrt(35)*(5*c1**3 - 3*c1)*s1*s2*c2/np.sqrt(np.pi),
	(621, 3, 0, 1, 1, 3, 1): lambda c1, c2, s1, s2: 0.328125*np.sqrt(2)*(5*c1**3 - 3*c1)*(5*c2**2 - 1)*s1*s2/np.sqrt(np.pi),
	(628, 3, 1, 1, -1, 2, -2): lambda c1, c2, s1, s2: 0.0234375*np.sqrt(210)*(5*c1**2 - 1)*s1**2*s2**2/np.sqrt(np.pi),
	(634, 3, 1, 1, -1, 3, -2): lambda c1, c2, s1, s2: 0.1640625*np.sqrt(30)*(5*c1**2 - 1)*s1**2*s2**2*c2/np.sqrt(np.pi),
	(643, 3, 1, 1, 0, 1, 1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(42)*(5*c1**2 - 1)*s1*s2*c1/np.sqrt(np.pi),
	(647, 3, 1, 1, 0, 2, 1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(210)*(5*c1**2 - 1)*s1*s2*c1*c2/np.sqrt(np.pi),
	(653, 3, 1, 1, 0, 3, 1): lambda c1, c2, s1, s2: 0.328125*np.sqrt(3)*(5*c1**2 - 1)*(5*c2**2 - 1)*s1*s2*c1/np.sqrt(np.pi),
	(656, 3, 1, 1, 1, 0, 0): lambda c1, c2, s1, s2: 0.09375*np.sqrt(14)*(5*c1**2 - 1)*s1**2/np.sqrt(np.pi),
	(658, 3, 1, 1, 1, 1, 0): lambda c1, c2, s1, s2: 0.09375*np.sqrt(42)*(5*c1**2 - 1)*s1**2*c2/np.sqrt(np.pi),
	(662, 3, 1, 1, 1, 2, 0): lambda c1, c2, s1, s2: 0.046875*np.sqrt(70)*(5*c1**2 - 1)*(3*c2**2 - 1)*s1**2/np.sqrt(np.pi),
	(664, 3, 1, 1, 1, 2, 2): lambda c1, c2, s1, s2: 0.0234375*np.sqrt(210)*(5*c1**2 - 1)*s1**2*s2**2/np.sqrt(np.pi),
	(668, 3, 1, 1, 1, 3, 0): lambda c1, c2, s1, s2: 0.328125*np.sqrt(2)*(5*c1**2 - 1)*(5*c2**3 - 3*c2)*s1**2/np.sqrt(np.pi),
	(670, 3, 1, 1, 1, 3, 2): lambda c1, c2, s1, s2: 0.1640625*np.sqrt(30)*(5*c1**2 - 1)*s1**2*s2**2*c2/np.sqrt(np.pi),
	(673, 3, 2, 1, -1, 1, -1): lambda c1, c2, s1, s2: -0.09375*np.sqrt(105)*s1**3*s2*c1/np.sqrt(np.pi),
	(677, 3, 2, 1, -1, 2, -1): lambda c1, c2, s1, s2: -0.46875*np.sqrt(21)*s1**3*s2*c1*c2/np.sqrt(np.pi),
	(681, 3, 2, 1, -1, 3, -3): lambda c1, c2, s1, s2: 0.8203125*np.sqrt(2)*s1**3*s2**3*c1/np.sqrt(np.pi),
	(683, 3, 2, 1, -1, 3, -1): lambda c1, c2, s1, s2: -0.1640625*np.sqrt(30)*(5*c2**2 - 1)*s1**3*s2*c1/np.sqrt(np.pi),
	(696, 3, 2, 1, 0, 2, 2): lambda c1, c2, s1, s2: 0.46875*np.sqrt(21)*s1**2*s2**2*c1**2/np.sqrt(np.pi),
	(702, 3, 2, 1, 0, 3, 2): lambda c1, c2, s1, s2: 3.28125*np.sqrt(3)*s1**2*s2**2*c1**2*c2/np.sqrt(np.pi),
	(707, 3, 2, 1, 1, 1, 1): lambda c1, c2, s1, s2: 0.09375*np.sqrt(105)*s1**3*s2*c1/np.sqrt(np.pi),
	(711, 3, 2, 1, 1, 2, 1): lambda c1, c2, s1, s2: 0.46875*np.sqrt(21)*s1**3*s2*c1*c2/np.sqrt(np.pi),
	(717, 3, 2, 1, 1, 3, 1): lambda c1, c2, s1, s2: 0.1640625*np.sqrt(30)*(5*c2**2 - 1)*s1**3*s2*c1/np.sqrt(np.pi),
	(719, 3, 2, 1, 1, 3, 3): lambda c1, c2, s1, s2: 0.8203125*np.sqrt(2)*s1**3*s2**3*c1/np.sqrt(np.pi),
	(724, 3, 3, 1, -1, 2, -2): lambda c1, c2, s1, s2: -0.1171875*np.sqrt(14)*s1**4*s2**2/np.sqrt(np.pi),
	(730, 3, 3, 1, -1, 3, -2): lambda c1, c2, s1, s2: -0.8203125*np.sqrt(2)*s1**4*s2**2*c2/np.sqrt(np.pi),
	(751, 3, 3, 1, 0, 3, 3): lambda c1, c2, s1, s2: 0.546875*np.sqrt(3)*s1**3*s2**3*c1/np.sqrt(np.pi),
	(760, 3, 3, 1, 1, 2, 2): lambda c1, c2, s1, s2: 0.1171875*np.sqrt(14)*s1**4*s2**2/np.sqrt(np.pi),
	(766, 3, 3, 1, 1, 3, 2): lambda c1, c2, s1, s2: 0.8203125*np.sqrt(2)*s1**4*s2**2*c2/np.sqrt(np.pi),
}


NUMSK = len(INTEGRALS_DIPOLE)

def convert_quant_num(l):
    """convert quantum number l to letter for string matching in select_subshells"""
    if l == 0:
        return 's'
    elif l == 1:
        return 'p'
    elif l == 2:
        return 'd'
    elif l ==3:
        return 'f'
    else:
        raise ValueError("invalid quantum number for angular momentum")
        
def convert_sk_index(lm_tuple):
    """
    Convert (l1, m1, l2, m2, l3, m3) into a string like 'sspxd1 for printing'.
    """
    if (len(lm_tuple)-1) % 2 != 0:
        raise ValueError("Tuple must have pairs of (l, m) quantum numbers.")
    
    # mapping from l to orbital letter
    l_map = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
    # mapping for p orbitals
    p_map = {-1: 'y', 0: 'z', 1: 'x'}
    # mapping for d orbitals (numbered)
    d_map = {-2: '1', -1: '2', 0: '3', 1: '4', 2: '5'}
    # mapping for f orbitals (numbered)
    f_map = {-3: '1', -2: '2', -1: '3', 0: '4', 1: '5', 2: '6', 3: '7'}
    
    out = []
    for i in range(0, len(lm_tuple)-1, 2):
        l, m = lm_tuple[i+1], lm_tuple[i+2]
        if l not in l_map:
            raise ValueError(f"Unsupported l={l}")
        l_char = l_map[l]
        
        if l == 0:
            part = 's' + 's'  # always "ss"
        elif l == 1:
            part = 'p' + p_map.get(m, '?')
        elif l == 2:
            part = 'd' + d_map.get(m, '?')
        elif l == 3:
            part = 'f' + f_map.get(m, '?')
        else:
            part = l_char  # fallback
        out.append(part)
    
    return ''.join(out)


def phi3(c1, c2, s1, s2, sk_label): 
    """ 
    Returns the angle-dependent part of the given two-center dipole-integral,
    with c1 and s1 (c2 and s2) the sine and cosine of theta_1 (theta_2)
    for the atom at origin (atom at z=Rz). These expressions are obtained
    by integrating analytically over phi.
    """
    return INTEGRALS_DIPOLE[sk_label](c1,c2,s1,s2)

def select_integrals(e1, e2):
    """
    Return list of integrals (integral, nl1, nl2)
    to be done for element pair e1, e2. nl1 and nl2 strings for 
    compatibility with existing hotcent code
    """
    selected = []
    for ival1, valence1 in enumerate(e1.basis_sets):
        for ival2, valence2 in enumerate(e2.basis_sets):
            for sk_label, func in INTEGRALS_DIPOLE.items():
                nl1, nl2 = select_subshells(valence1, valence2, sk_label)
                if nl1 is not None and nl2 is not None:
                    selected.append((sk_label, nl1, nl2))
    return selected

def select_subshells(val1, val2, sk_label):
    """
    Select subshells from given valence sets to calculate given
    Slater-Koster integral.

    Parameters
    ----------
    val1, val2 : list of str
        Valence subshell sets (e.g. ['2s', '2p'], ['4s', '3d']).
    integral : str
        Slater-Koster integral label (e.g. 'pzpzpz').

    Returns
    -------
    nl1, nl2 : str
        Matching subshell pair (e.g. ('2s', '3d') in this example).
    """
    nl1 = None
    for nl in val1:
        if nl[1] == convert_quant_num(sk_label[1]):
            nl1 = nl

    nl2 = None
    for nl in val2:
        if nl[1] == convert_quant_num(sk_label[5]):
            nl2 = nl

    return nl1, nl2
    
    
def print_integral_overview(e1, e2, selected, file):
    """ Prints an overview of the selected Slater-Koster integrals. """
    for bas1 in range(len(e1.basis_sets)):
        for bas2 in range(len(e2.basis_sets)):
            sym1 = e1.get_symbol() + '+'*bas1
            sym2 = e2.get_symbol() + '+'*bas2
            print('Integrals for %s-%s pair:' % (sym1, sym2), end=' ',
                  file=file)
            for integral, nl1, nl2 in selected:
                if e1.get_basis_set_index(nl1) == bas1 and \
                   e2.get_basis_set_index(nl2) == bas2:
                    print('_'.join([nl1, nl2, convert_sk_index(integral)]), end=' ', file=file)
            print(file=file, flush=True)
    return

def tail_smoothening(x, y_in, eps_inner=1e-8, eps_outer=1e-16, window_size=5):
    """ Smoothens the tail for the given function y(x).

    Parameters
    ----------
    x : np.array
        Array with grid points (strictly increasing).
    y_in : np.array
        Array with function values.
    eps_inner : float, optional
        Inner threshold. Tail values with magnitudes between this value and
        the outer threshold are subjected to moving window averaging to
        reduce noise.
    eps_outer : float, optional
        Outer threshold. Tail values with magnitudes below this value
        are set to zero.
    window_size : int, optional
        Moving average window size (odd integers only).

    Returns
    -------
    y_out : np.array
        Array with function values with a smoothed tail.
    """
    assert window_size % 2 == 1, 'Window size needs to be odd.'

    y_out = np.copy(y_in)
    N = len(y_out)

    if np.all(abs(y_in) < eps_outer):
        return y_out

    Nzero = 0
    izero = -1
    for izero in range(N-1, 1, -1):
        if abs(y_out[izero]) < eps_outer:
            Nzero += 1
        else:
            break

    y_out[izero+1:] = 0.

    Nsmall = 0
    for ismall in range(izero, 1, -1):
        if abs(y_out[ismall]) < eps_inner:
            Nsmall += 1
        else:
            break
    else:
        ismall -= 1

    if Nsmall > 0:
        tail = np.empty(Nsmall-1)
        half = (window_size - 1) // 2
        for j, i in enumerate(range(ismall+1, izero)):
            tail[j] = np.mean(y_out[i-half:i+half+1])

        y_out[ismall+1:izero] = tail

    return y_out
    

def write_skf(handle, Rgrid, table, has_atom_transition, mass, atom_transitions):
    """
    Writes a parameter file in '.skf' format starting at grid_dist 
    and giving nonzero values from R_grid[0] on.

    Parameters
    ----------
    handle : file handle
        Handle of an open file.
    Rgrid : list or array
        Lists with interatomic distances.
    table : nd.ndarray
        Two-dimensional array with the Slater-Koster table.

    Other parameters
    ----------------
    See Offsite2cTable.write()
    """
    # TODO find out what all the other quantities are, that do not come from table

    grid_dist = Rgrid[1] - Rgrid[0]
    grid_npts, numint = np.shape(table)
    assert (numint % NUMSK) == 0
    nzeros = int(np.round(Rgrid[0] / grid_dist)) - 1
    assert nzeros >= 0
    print("%.12f, %d" % (grid_dist, grid_npts + nzeros), file=handle)

    keys_sorted = sorted(INTEGRALS_DIPOLE.keys(), key= lambda x: x[0])
    if has_atom_transition:
        atom_integrals = [atom_transitions[i] for i in keys_sorted]
        print(" ".join(f"{x}" for x in atom_integrals), file=handle)

    print("%.3f, 19*0.0" % mass, file=handle) # TODO change number of columns

    # Table containing the Slater-Koster integrals
    numtab = numint // NUMSK
    assert numtab > 0
    
    indices = np.shape(table)[1]
    for i in range(nzeros):
        line = ''
        for j in range(indices):
                line += '{0: 1.12e}  '.format(0) 
        print(line, file=handle)
    for i in range(grid_npts):
        line = ''
        for j in range(indices):
                line += '{0: 1.12e}  '.format(table[i, j])
        print(line, file=handle)
    
    